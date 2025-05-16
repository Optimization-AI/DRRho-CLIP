import json
import logging
import math
import os
import time
import contextlib
import pathlib
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

try:
    import wandb
except ImportError:
    wandb = None

from fast_clip import get_input_dtype
from .distributed import is_master, all_gather_tuple_tensor
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


@torch.no_grad()
def sync_model(model, ref_model):
    """Sync model weights with a reference model"""
    model = unwrap_model(model)
    ref_model = unwrap_model(ref_model)
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        param.data.copy_(ref_param.data.to(param.device))


@torch.no_grad()
def shard_features(
        features: torch.Tensor,
        offset: int,
        save_dir: str | pathlib.Path,
        format: str = "{:08d}.pt",
        num_samples_per_shard: int = 10000,
        shards_id_list = None,
        ):
    save_dir = pathlib.Path(save_dir)
    assert offset % num_samples_per_shard == 0
    starting_shard = offset // num_samples_per_shard
    num_shards = (features.shape[0] + num_samples_per_shard - 1) // num_samples_per_shard
    logging.info(f"Sharding {features.shape[0]} entries into {num_shards} shards")
    for i in range(num_shards):
        shard = features[i * num_samples_per_shard: min((i + 1) * num_samples_per_shard, features.shape[0])].clone()
        shard_path = save_dir / format.format(shards_id_list[i] if shards_id_list is not None else i + starting_shard)
        torch.save(shard, shard_path)


@torch.no_grad()
def cache_features(ref_model, ref_features_dict, data, args, num_samples_per_shard: int = 10000):
    ref_features_offset = args.ref_features_offset
    device = torch.device(args.device)
    ref_model.eval()
    ref_features_all = ref_features_dict["features"]
    is_ref_features_cached = ref_features_dict["is_cached"]

    data['train'].set_epoch(0)
    dataloader = data['train'].dataloader
    num_batches = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    shards_id_list = torch.tensor(data['train'].shards_id_list, dtype=torch.int64) - ref_features_offset // num_samples_per_shard
    shards_id_real_to_compact = torch.zeros(shards_id_list[-1] + 1, dtype=torch.int64) - 1
    shards_id_real_to_compact[shards_id_list] = torch.arange(len(shards_id_list))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        # we assume image indices and text indices are the same
        images, texts, idx, _ = batch
        idx = idx - ref_features_offset
        shards_id_compact = shards_id_real_to_compact[idx // num_samples_per_shard]
        assert -1 not in shards_id_compact
        idx_compact = idx % num_samples_per_shard + shards_id_compact * num_samples_per_shard
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)

        ref_model_out = ref_model(images, texts)
        ref_features = [ref_model_out["image_features"], ref_model_out["text_features"]]
        ref_features_all[idx_compact, 0] = ref_features[0].to("cpu")
        ref_features_all[idx_compact, 1] = ref_features[1].to("cpu")
        is_ref_features_cached[idx_compact] = True

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % args.log_every_n_steps == 0 or batch_count == num_batches:
            batch_size = len(images)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches

            samples_per_second = args.batch_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            logging.info(
                f"[Rank {args.rank:>2d}] Reference Features Caching: "
                f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                f"{samples_per_second_per_gpu:#g}/s/gpu "
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


@torch.no_grad()
def cache_syn_texts(cap_model, syn_texts_dict, data, args):
    syn_texts_offset = args.syn_texts_offset
    num_generated = args.num_syn_texts
    input_dtype = get_input_dtype(args.precision)
    device = torch.device(args.device)
    syn_texts_all = syn_texts_dict["syn_texts"]
    is_syn_texts_cached = syn_texts_dict["is_cached"]

    data['train'].set_epoch(0)
    dataloader = data['train'].dataloader
    num_batches = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        # we assume image indices and text indices are the same
        images, _, idx, _ = batch
        idx = idx - syn_texts_offset
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        data_time_m.update(time.time() - end)

        syn_texts_stack = []
        for _ in range(num_generated):
            with torch.cuda.amp.autocast():
                syn_texts = cap_model.generate(images, seq_len=77, temperature=0.75,
                                               generation_type='top_k', top_k=50, fixed_output_length=True)
            syn_texts_stack.append(syn_texts)
        syn_texts = torch.stack(syn_texts_stack).movedim(0, 1)
        syn_texts_all[idx] = syn_texts.to("cpu")
        is_syn_texts_cached[idx] = True

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % args.log_every_n_steps == 0 or batch_count == num_batches:
            batch_size = len(images)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches

            samples_per_second = args.batch_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            logging.info(
                f"[Rank {args.rank:>2d}] Synthetic Captions Generation: "
                f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                f"{samples_per_second_per_gpu:#g}/s/gpu "
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


# https://arxiv.org/abs/2406.17711
def softmax_neg(logits, is_sampled=None, dim=-1):
    # during the iterative sampling, we need to mask out negatives that have already been selected
    if is_sampled is not None:
        logits_neg = logits - (1.0 - is_sampled) * 1e8
    else:
        logits_neg = logits
    logits_neg = torch.logsumexp(logits_neg, dim=dim)       # Compute the softmax negative term
    return logits_neg


# https://arxiv.org/abs/2406.17711
def softmax_nll(params, embeds, is_sampled_all=None, embeds_all=None):
    zimg, ztxt = embeds
    if embeds_all is None:
        embeds_all = embeds
    zimg_all, ztxt_all = embeds_all
    logits_mat = zimg @ ztxt.T * params["t"]
    logits_mat_all_img = zimg @ ztxt_all.T * params["t"]
    logits_mat_all_txt = ztxt @ zimg_all.T * params["t"]
    if is_sampled_all is not None:
        is_sampled_all = is_sampled_all.reshape(1, logits_mat_all_img.shape[1])
    # calculate negative softmax term for image to text half of loss
    logits_ij = softmax_neg(logits_mat_all_img, is_sampled_all, dim=1)
    # calculate negative softmax term for text to image half of loss
    logits_ji = softmax_neg(logits_mat_all_txt, is_sampled_all, dim=1)
    loss_0 = -(torch.diag(logits_mat) - logits_ij)
    loss_1 = -(torch.diag(logits_mat) - logits_ji)
    neg_logits = 0.5 * (logits_ij + logits_ji)
    l = torch.mean(0.5 * (loss_0 + loss_1))
    return l, neg_logits, -logits_mat


# https://arxiv.org/abs/2406.17711
def jointly_sample_batch_softmax(embeds_ref, embeds_learner, params_ref, params_learner,
                                 n_chunks=2, filter_ratio=0.8, topk=False):
    if torch.distributed.is_initialized():
        embeds_ref_all = [torch.cat(torch.distributed.nn.all_gather(embeds_ref[0]), dim=0),
                          torch.cat(torch.distributed.nn.all_gather(embeds_ref[1]), dim=0)]
        embeds_learner_all = [torch.cat(torch.distributed.nn.all_gather(embeds_learner[0]), dim=0),
                              torch.cat(torch.distributed.nn.all_gather(embeds_learner[1]), dim=0)]
    else:
        embeds_ref_all = embeds_ref
        embeds_learner_all = embeds_learner
    softmax_score_gain = 1.0
    _, _, learner_logits = softmax_nll(params_learner, embeds_learner, is_sampled_all=None,
                                       embeds_all=embeds_learner_all)
    _, _, ref_logits = softmax_nll(params_ref, embeds_ref, is_sampled_all=None, embeds_all=embeds_ref_all)
    scores = (learner_logits - ref_logits) * softmax_score_gain
    n_images = scores.shape[0]                                  # scores.shape = [B, B]
    n_draws = int(n_images * (100 - filter_ratio * 100) / 100 / n_chunks)     # Size of each chunk.
    logits_ii = torch.diag(scores)                              # Self-similarity scores.
    inds = (logits_ii - logits_ii.min()).multinomial(n_draws)   # Subtract minimum value to avoid negative weight

    for _ in range(n_chunks - 1):
        # Binary indicator of current samples [n_images,].
        is_sampled = torch.eye(n_images, device=scores.device)[inds].sum(dim=0)
        is_sampled_all = torch.cat(torch.distributed.nn.all_gather(is_sampled))
        _, learner_logits_n, _ = softmax_nll(params_learner, embeds_learner, is_sampled_all=is_sampled_all,
                                             embeds_all=embeds_learner_all)
        _, ref_logits_n, _ = softmax_nll(params_ref, embeds_ref, is_sampled_all=is_sampled_all,
                                         embeds_all=embeds_ref_all)
        rho_scores_n = (learner_logits_n - ref_logits_n) * softmax_score_gain
        logits = logits_ii + rho_scores_n                       # Conditional learnability given past samples.
        logits = logits - is_sampled * 1e8                      # Avoid sampling with replacement.
        if topk:
            new_inds = torch.exp(logits).topk(n_draws).indices
        else:
            new_inds = torch.exp(logits).multinomial(n_draws)
        inds = torch.concatenate((inds, new_inds))              # Expand the array of indices sampled.
    return inds.cpu()                                           # Gather and return subset indices.


# https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/utils/clip_grad.py
@torch.no_grad()
def get_grad_norm(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])

    norms = []
    for ((device, _), ([grads], _)) in grouped_grads.items():
        if _has_foreach_support(grads, device=device):
            norms.extend(torch._foreach_norm(grads, norm_type))
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    return total_norm


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None,
                    profiler=None, ref_model=None, cap_model=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    debug_stats = None

    model.train()
    if args.distill:
        dist_model.eval()

    if args.fastclip:
        loss.adjust_hyperparams(epoch)
    no_sync_mgr = contextlib.nullcontext
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
        if torch.distributed.get_world_size() == 1 and isinstance(model, torch.nn.parallel.DistributedDataParallel):
            no_sync_mgr = model.no_sync
    offset = rank * args.train_batch_size

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if ref_model is not None:
        assert args.accum_freq == 1, "Reference model only supported with accum_freq=1"
        ref_model.eval()

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if args.stop_iters > 0 and step >= args.stop_iters:
            break

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, image_idx, text_idx = batch[:4]
        batch_iter = iter(batch[4:])
        if args.cached_ref_features_dir:
            ref_features = next(batch_iter)
        else:
            ref_features = None
        if args.cached_syn_texts_dir:
            syn_texts = next(batch_iter)                                    # [batch, num_syn_texts, seq_len]
        elif cap_model is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                syn_texts = cap_model.generate(images, seq_len=77, temperature=0.75,
                                               generation_type='top_k', top_k=50, fixed_output_length=True)
                syn_texts = syn_texts.unsqueeze(1)
        else:
            syn_texts = None
        if syn_texts is not None:
            # texts_all = torch.cat((texts.unsqueeze(1), syn_texts), dim=1)   # [batch, num_syn_texts + 1, seq_len]
            texts_all = syn_texts
            texts = texts_all[torch.arange(texts_all.shape[0]),
                              torch.randint(0, texts_all.shape[1], (texts_all.shape[0],))]

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        indices = [image_idx, text_idx]
        image_idx = image_idx.unsqueeze(-1).to(device=device, dtype=image_idx.dtype, non_blocking=True)
        text_idx = text_idx.unsqueeze(-1).to(device=device, dtype=text_idx.dtype, non_blocking=True)
        indices_device = [image_idx, text_idx]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            if ref_features is None:
                if ref_model is not None:
                    with torch.no_grad():
                        ref_model_out = ref_model(images, texts)
                        ref_features = [ref_model_out["image_features"], ref_model_out["text_features"]]
            else:
                # [batch, image/text, feature]
                ref_features = ref_features.movedim(1, 0)
                if syn_texts is not None and ref_model is not None:
                    with torch.no_grad():
                        ref_model_out = ref_model(None, texts)
                        ref_features[1] = ref_model_out["text_features"]
            if ref_features is not None:
                ref_features = [ref_features[0].to(device), ref_features[1].to(device)]

            if "select" in args.ref_features_usage:
                assert ref_features is not None, "Reference features must be provided for selection"
                with torch.no_grad():
                    model_out = model(images, texts)
                    features = [model_out["image_features"], model_out["text_features"]]
                    logit_scale = model_out["logit_scale"]
                    inds = jointly_sample_batch_softmax(ref_features, features, {"t": 100.0},
                                                        {"t": logit_scale}, filter_ratio=args.ref_filter_ratio,
                                                        n_chunks=args.ref_filter_n_chunks, topk=args.ref_filter_topk)
                images, texts = images[inds], texts[inds]
                indices = [indices[0][inds], indices[1][inds]]
                indices_device = [indices_device[0][inds], indices_device[1][inds]]
                if "ref" not in args.ref_features_usage:
                    ref_features = None
                else:
                    ref_features = [ref_features[0][inds], ref_features[1][inds]]

            with no_sync_mgr():
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    if args.distill:
                        if ref_features is not None:
                            model_out.update({"dist_image_features": ref_features[0], "dist_text_features": ref_features[1]})
                        else:
                            with torch.no_grad():
                                dist_model_out = dist_model(images, texts)
                            model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    if args.fastclip:
                        features = [model_out["image_features"], model_out["text_features"]]
                        model_out = {"features": features, "indices": indices, "logit_scale": logit_scale,
                                     "offset": offset, "ref_features": ref_features}
                        if "ref" in args.ref_features_usage and "distill" in args.ref_features_usage \
                            and args.distill_ref_logit_scale > 0:
                                model_out.update({"logit_scale": torch.tensor(args.distill_ref_logit_scale)})
                        if "distill" in args.ref_features_usage:
                            model_out.update({"dist_features": ref_features,
                                              "dist_loss_weight": args.distill_weight,
                                              "dist_student_logit_scale": logit_scale,
                                              "dist_teacher_logit_scale": args.distill_logit_scale})
                    losses = loss(**model_out, output_dict=True)
                    if args.fastclip:
                        gather_list = losses["gather_list"]
                        del losses["gather_list"]
                        remote_gather_list = all_gather_tuple_tensor(gather_list, None)
                        remote_indices = all_gather_tuple_tensor(indices_device, None)
                        remote_indices[0] = remote_indices[0].squeeze(-1).to(device="cpu", dtype=torch.int64)
                        remote_indices[1] = remote_indices[1].squeeze(-1).to(device="cpu", dtype=torch.int64)
                        loss.set_params(*remote_indices, *remote_gather_list)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
        else:
            with torch.no_grad(), autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                features_all = [model_out["image_features"], model_out["text_features"]]
                if args.fastclip:
                    remote_features = all_gather_tuple_tensor(features_all)
                    # update u, etc
                    local_args = {"features": features_all, "indices": indices, "remote_features": remote_features,
                                    "logit_scale": logit_scale, "offset": offset}
                    _, _, _, _, gather_list = loss.local(**local_args)
                    gather_list = indices_device + gather_list
                    remote_gather_list = all_gather_tuple_tensor(gather_list, None)
                    remote_gather_list[0] = remote_gather_list[0].squeeze(-1).to(device="cpu", dtype=torch.int64)
                    remote_gather_list[1] = remote_gather_list[1].squeeze(-1).to(device="cpu", dtype=torch.int64)
                    loss.set_params(*remote_gather_list)
                    remote_gather_list = all_gather_tuple_tensor(gather_list)
                    remote_u = remote_gather_list[2:4]
                    if "individual" in args.temperature_scheme:
                        remote_tau = remote_gather_list[4:6]
                        remote_bounds = remote_gather_list[6:8]

            micro_batch_size = args.batch_size // args.accum_freq
            for j in range(args.accum_freq):
                images_micro = images[j * micro_batch_size:(j + 1) * micro_batch_size]
                texts_micro = texts[j * micro_batch_size:(j + 1) * micro_batch_size]
                indices_micro = [indices[0][j * micro_batch_size:(j + 1) * micro_batch_size],
                                 indices[1][j * micro_batch_size:(j + 1) * micro_batch_size]]
                with no_sync_mgr():
                    with autocast():
                        model_out = model(images_micro, texts_micro)
                        logit_scale = model_out["logit_scale"]
                        if args.fastclip:
                            features_micro = [model_out["image_features"], model_out["text_features"]]
                            offset_micro = offset + j * micro_batch_size
                            local_args = {"features": features_micro, "indices": indices_micro,
                                          "remote_features": remote_features, "logit_scale": logit_scale,
                                          "offset": offset_micro}
                            loss1_im, loss1_tt, sim_im, sim_tt, _ = loss.local(**local_args)
                            # here sim_im is local_im vs. global_tt, sim_tt is local_tt vs. global_im
                            sim = (sim_tt.T, sim_im.T)
                            u = [loss.get_params(indices_micro[0], loss.u_im)[0],
                                 loss.get_params(indices_micro[1], loss.u_tt)[0]]
                            if "individual" in args.temperature_scheme:
                                model_out.update({"remote_tau": remote_tau, "remote_bounds": remote_bounds})
                            model_out.update(
                                {"features": features_micro, "remote_features": remote_features, "remote_u": remote_u})
                            model_out.update(
                                {"offset": offset_micro, "loss1": (loss1_im, loss1_tt), "u": u, "sim": sim})
                        else:
                            image_features, text_features = features_all[0].clone(), features_all[1].clone()
                            image_features[j * micro_batch_size:(j + 1) * micro_batch_size] = model_out["image_features"]
                            text_features[j * micro_batch_size:(j + 1) * micro_batch_size] = model_out["text_features"]
                            model_out.update({"image_features": image_features, "text_features": text_features})
                        losses = loss(**model_out, output_dict=True)

                        total_loss = sum(losses.values()) / args.accum_freq
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(args.logit_scale_bound))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and args.debug:
            grad_norm = get_grad_norm(unwrap_model(model).parameters()).item()
            if debug_stats is None:
                debug_stats = {}
            if "grad_norm" not in debug_stats.keys():
                debug_stats["grad_norm"] = []
            debug_stats["grad_norm"].append(grad_norm)
            if "index" not in debug_stats.keys():
                debug_stats["index"] = []
            debug_stats["index"].append(image_idx)
            for key, val in losses.items():
                if key not in debug_stats.keys():
                    debug_stats[key] = []
                debug_stats[key].append(val.item())
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = args.batch_size
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            if args.fastclip and "individual" in args.temperature_scheme:
                lr_tau = loss.lr_tau
            else:
                lr_tau = optimizer.param_groups[-1]['lr']
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"ETA: {datetime.timedelta(seconds=int((args.training_steps - step) * batch_time_m.avg))} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} LR_tau: {lr_tau:.5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            if args.debug:
                torch.save(debug_stats, os.path.join(args.debug_path, f"iter_{step}.pt"))
                debug_stats = None

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if profiler is not None:
            profiler.step()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {"epoch": epoch}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.logs, args.name, f"eval_results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
