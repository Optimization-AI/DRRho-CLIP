import math
import logging
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        if not has_distributed or not dist.is_initialized():
            self.rank = 0
            self.world_size = 1
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


def dot_ensemble_features(feat_a, feat_b, logit_scale, dims):
    """Compute sum_t Softmax(a_t @ b_t) for between features from an ensemble model."""
    num_members = len(dims)
    dims = np.cumsum([0] + dims)
    logits = [
        logit_scale * (feat_a[:, dims[i]:dims[i+1]] @ feat_b[dims[i]:dims[i+1], :])
        for i in range(num_members)
    ]
    logits = sum([F.softmax(logit, dim=1) for logit in logits]) / num_members
    return logits


class DistillClipLoss(ClipLoss):

    def __init__(
        self,
        *args,
        teacher_dimension=[-1],
        distill_loss_weights=[1.0, 1.0],
        average_after_softmax=False,
        dist_logit_scale=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dist_logit_scale = dist_logit_scale
        self.teacher_dimension = teacher_dimension
        self.distill_loss_weights = distill_loss_weights
        self.average_after_softmax = average_after_softmax

    def get_logits_dist(self, image_features, text_features, logit_scale):
        dims = self.teacher_dimension
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = dot_ensemble_features(image_features, all_text_features.T, logit_scale, dims)
                logits_per_text = dot_ensemble_features(text_features, all_image_features.T, logit_scale, dims)
            else:
                logits_per_image = dot_ensemble_features(all_image_features, all_text_features.T, logit_scale, dims)
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = dot_ensemble_features(image_features, text_features.T, logit_scale, dims)
            logits_per_text = dot_ensemble_features(text_features, image_features.T, logit_scale, dims)

        return logits_per_image, logits_per_text

    def dist_loss(self, teacher_logits, student_logits):
        if self.average_after_softmax:
            return -(teacher_logits * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
        else:
            return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale=None,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        if self.dist_logit_scale is not None:
            dist_logit_scale = self.dist_logit_scale

        if self.average_after_softmax:
            dist_logits_per_image, dist_logits_per_text = \
                self.get_logits_dist(dist_image_features, dist_text_features, dist_logit_scale)
        else:
            dist_logits_per_image, dist_logits_per_text = \
                self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2 * self.distill_loss_weights[0]

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2 * self.distill_loss_weights[1]

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class FastCLIPLoss(nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma: float,
                 gamma_schedule: str = "constant",
                 gamma_decay_epochs: int = -1,
                 rho: float = 8.0,
                 eps: float = 1e-14,
                 multiply_tau: bool = True,
                 cache_mask: bool = True,
                 device: torch.device = torch.device("cuda"),
                 ):
        """Create an instance of Global Contrastive Loss with global temperature parameter."""
        super(FastCLIPLoss, self).__init__()
        self.data_size = data_size
        self.gamma = 1.0
        self.gamma_orig = gamma
        self.gamma_schedule = gamma_schedule
        self.gamma_decay_epochs = gamma_decay_epochs
        if self.gamma_schedule != "none":
            assert self.gamma_decay_epochs > 0
        self.rho = rho
        self.eps = eps
        self.multiply_tau = multiply_tau
        self.cache_mask = cache_mask
        self.device = device

        self.u_im = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.u_tt = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.arange = {}
        self.mask = {}

        logging.info(f"data size: {data_size}, final gamma: {gamma}, gamma_schedule: {gamma_schedule}, "
                     f"gamma_decay_epochs: {self.gamma_decay_epochs}, rho: {rho}, eps: {self.eps}, "
                     f"multiply_tau: {self.multiply_tau}, cache_mask: {self.cache_mask}.")

    def adjust_gamma(self, epoch: int):
        if epoch == 0:
            self.gamma = 1.0
        elif epoch >= self.gamma_decay_epochs:
            self.gamma = self.gamma_orig
        else:
            self.gamma = self.gamma_orig
            if self.gamma_schedule == "cosine":
                self.gamma = 0.5 * (1 + math.cos(math.pi * epoch / self.gamma_decay_epochs)) * \
                             (1 - self.gamma_orig) + self.gamma_orig
        logging.info(f"gamma: {self.gamma}")

    def adjust_hyperparams(self, epoch: int):
        self.adjust_gamma(epoch)

    def get_params(self,
                   idx: Optional[Tensor] = None,
                   *args,
                   ):
        results = []
        for src in args:
            results.append(src[idx].to(self.device))
        return results

    def set_params(self,
                   image_idx: Optional[Tensor] = None, text_idx: Optional[Tensor] = None,
                   u_im: Optional[Tensor] = None, u_tt: Optional[Tensor] = None,
                   **kwargs
                   ):
        src_im_list = [u_im]
        dst_im_list = [self.u_im]
        src_tt_list = [u_tt]
        dst_tt_list = [self.u_tt]
        for src_im, dst_im, src_tt, dst_tt in zip(src_im_list, dst_im_list, src_tt_list, dst_tt_list):
            if src_im is not None:
                assert image_idx is not None and dst_im.device == image_idx.device
                dst_im[image_idx] = src_im.to("cpu")
            if src_tt is not None:
                assert text_idx is not None and dst_tt.device == text_idx.device
                dst_tt[text_idx] = src_tt.to("cpu")

    def get_arange(self, length: int, offset: int):
        # here we assume arange is on self.device
        # the arange should be small in size, so we force caching it
        if offset not in self.arange.keys():
            self.arange[offset] = {}
        if length not in self.arange[offset].keys():
            self.arange[offset][length] = torch.arange(length, device=self.device) + offset
        return self.arange[offset][length]

    def get_mask(self, height: int, width: int, offset: int):
        """Return a height * width matrix, with diagonal [offset: offset + height, offset: offset + height]
            being 0 and the rest being 1
        """
        if not self.cache_mask or (height, width, offset) not in self.mask.keys():
            mask_inv = torch.nn.functional.one_hot(self.get_arange(height, offset), width).to(self.device)
            mask = 1 - mask_inv
            if self.cache_mask and (height, width, offset) not in self.mask.keys():
                self.mask[(height, width, offset)] = (mask, mask_inv)
        else:
            mask, mask_inv = self.mask[(height, width, offset)]
        return mask, mask_inv

    def pairwise_loss(self,
                      features1: Tuple[Tensor, Tensor],
                      features2: Tuple[Tensor, Tensor],
                      logit_scale_im: Tensor,
                      sim: Optional[Tuple[Tensor, Tensor]] = None,
                      logit_scale_tt: Optional[Tensor] = None,
                      ref_features1: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_features2: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_sim: Optional[Tuple[Tensor, Tensor]] = None,
                      ):
        image_features1, text_features1 = features1[0], features1[1]
        image_features2, text_features2 = features2[0], features2[1]
        if logit_scale_tt is None:
            logit_scale_tt = logit_scale_im

        if sim is not None:
            sim_image, sim_text = sim[0], sim[1]
        else:
            sim_image = image_features1 @ text_features2.T  # shape [b1, b2]
            sim_text = text_features1 @ image_features2.T  # shape [b1, b2]
        diag_sim = torch.sum(torch.mul(image_features1, text_features1), dim=-1, keepdim=True)

        diff_image = (sim_image - diag_sim).mul(logit_scale_im)
        diff_text = (sim_text - diag_sim).mul(logit_scale_tt)
        if ref_features1 is not None:
            assert ref_features2 is not None
            ref_image_features1, ref_text_features1 = ref_features1[0], ref_features1[1]
            ref_image_features2, ref_text_features2 = ref_features2[0], ref_features2[1]
            if ref_sim is not None:
                ref_sim_image, ref_sim_text = ref_sim[0], ref_sim[1]
            else:
                ref_sim_image = ref_image_features1 @ ref_text_features2.T
                ref_sim_text = ref_text_features1 @ ref_image_features2.T
            ref_diag_sim = torch.sum(torch.mul(ref_image_features1, ref_text_features1), dim=-1, keepdim=True)
            diff_image = diff_image - (ref_sim_image - ref_diag_sim).mul(logit_scale_im)
            diff_text = diff_text - (ref_sim_text - ref_diag_sim).mul(logit_scale_tt)

        results = {"sim_image": sim_image, "sim_text": sim_text, "diff_image": diff_image,
                   "diff_text": diff_text, "diag_sim": diag_sim}
        return results

    def distill_loss(self,
                     student_features: Tuple[Tensor, Tensor],
                     teacher_features: Tuple[Tensor, Tensor],
                     student_logit_scale: Tensor | float,
                     teacher_logit_scale: Tensor | float | None = None,
                     student_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
                     teacher_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
                     ):
        if teacher_logit_scale is None:
            teacher_logit_scale = student_logit_scale
        if student_remote_features is None:
            if dist.is_initialized():
                student_remote_features = [torch.cat(torch.distributed.nn.all_gather(student_features[0]), dim=0),
                                           torch.cat(torch.distributed.nn.all_gather(student_features[1]), dim=0)]
            else:
                student_remote_features = student_features
        if teacher_remote_features is None:
            if dist.is_initialized():
                teacher_remote_features = [torch.cat(torch.distributed.nn.all_gather(teacher_features[0]), dim=0),
                                           torch.cat(torch.distributed.nn.all_gather(teacher_features[1]), dim=0)]
            else:
                teacher_remote_features = teacher_features

        student_logits_image = student_logit_scale * student_features[0] @ student_remote_features[1].T
        student_logits_text = student_logit_scale * student_features[1] @ student_remote_features[0].T
        teacher_logits_image = teacher_logit_scale * teacher_features[0] @ teacher_remote_features[1].T
        teacher_logits_text = teacher_logit_scale * teacher_features[1] @ teacher_remote_features[0].T

        dist_loss_image = -(teacher_logits_image.softmax(dim=1) * student_logits_image.log_softmax(dim=1)).sum(dim=1).mean()
        dist_loss_text = -(teacher_logits_text.softmax(dim=1) * student_logits_text.log_softmax(dim=1)).sum(dim=1).mean()
        dist_loss = (dist_loss_image + dist_loss_text) / 2

        return dist_loss

    def local(self,
              features: Tuple[Tensor, Tensor],
              indices: Tuple[Tensor, Tensor],
              remote_features: Tuple[Tensor, Tensor],
              logit_scale: Tensor,
              offset: int,
              ref_features: Optional[Tuple[Tensor, Tensor]] = None,
              ref_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
              **kwargs
              ):
        image_idx, text_idx = indices[0], indices[1]
        u_im = self.get_params(image_idx, self.u_im)[0]
        u_tt = self.get_params(text_idx, self.u_tt)[0]

        results = self.pairwise_loss(features, remote_features, logit_scale,
            ref_features1=ref_features, ref_features2=ref_remote_features)
        sim_im, sim_tt = results["sim_image"], results["sim_text"]
        diff_im, diff_tt = results["diff_image"], results["diff_text"]
        diag_sim = results["diag_sim"]

        # compute log(eps + g)
        with torch.no_grad():
            mask, mask_inv = self.get_mask(diff_im.shape[0], diff_im.shape[1], offset)
            diff_im_shifted = diff_im + mask_inv * math.log(self.eps * diff_im.shape[1])
            diff_tt_shifted = diff_tt + mask_inv * math.log(self.eps * diff_tt.shape[1])
            logsumexp_im = torch.logsumexp(diff_im_shifted - diag_sim - math.log(self.eps * diff_im.shape[1]),
                                           dim=-1, keepdim=True) + math.log(self.eps)
            logsumexp_tt = torch.logsumexp(diff_tt_shifted - diag_sim - math.log(self.eps * diff_tt.shape[1]),
                                           dim=-1, keepdim=True) + math.log(self.eps)
        if self.gamma == 1.0:
            u_im = logsumexp_im
            u_tt = logsumexp_tt
        else:
            bad_im_idx = torch.nonzero(
                (u_im == 0.0).logical_or(u_im.isinf()).logical_or(u_im.isnan()), as_tuple=True)[0]
            bad_tt_idx = torch.nonzero(
                (u_tt == 0.0).logical_or(u_tt.isinf()).logical_or(u_tt.isnan()), as_tuple=True)[0]
            u_diff_im = u_im + math.log(1- self.gamma) - (logsumexp_im + math.log(self.gamma))
            u_diff_tt = u_tt + math.log(1- self.gamma) - (logsumexp_tt + math.log(self.gamma))
            u_im = torch.where(u_diff_im > 0,
                               u_im + math.log(1- self.gamma) + torch.log(1 + torch.exp(-u_diff_im)),
                               logsumexp_im + math.log(self.gamma) + torch.log(1 + torch.exp(u_diff_im)))
            u_tt = torch.where(u_diff_tt > 0,
                               u_tt + math.log(1- self.gamma) + torch.log(1 + torch.exp(-u_diff_tt)),
                               logsumexp_tt + math.log(self.gamma) + torch.log(1 + torch.exp(u_diff_tt)))
            if bad_im_idx.shape[0] > 0:
                u_im[bad_im_idx] = logsumexp_im[bad_im_idx].to(u_im.dtype)
            if bad_tt_idx.shape[0] > 0:
                u_tt[bad_tt_idx] = logsumexp_tt[bad_tt_idx].to(u_tt.dtype)

        with torch.no_grad():
            weight_im = torch.exp(diff_im - u_im)
            weight_tt = torch.exp(diff_tt - u_tt)
        loss1_im = torch.sum(diff_im * weight_im * mask, dim=-1, keepdim=True) / (diff_im.shape[1] - 1)
        loss1_tt = torch.sum(diff_tt * weight_tt * mask, dim=-1, keepdim=True) / (diff_tt.shape[1] - 1)

        gather_list = [u_im, u_tt]

        return loss1_im, loss1_tt, sim_im, sim_tt, gather_list

    def forward(self,
                features: Tuple[Tensor, Tensor],
                remote_features: Tuple[Tensor, Tensor],
                remote_u: Tuple[Tensor, Tensor],
                loss1: Tuple[Tensor, Tensor],
                u: Tuple[Tensor, Tensor],
                sim: Tuple[Tensor, Tensor],
                logit_scale: Tensor,
                offset: int,
                output_dict: bool = False,
                loss_weight: float = 1.0,
                ref_features: Optional[Tuple[Tensor, Tensor]] = None,
                ref_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
                dist_features: Optional[Tuple[Tensor, Tensor]] = None,
                dist_loss_weight: float = 1.0,
                dist_student_logit_scale: Tensor | None = None,
                dist_teacher_logit_scale: Tensor | float = 100.0,
                **kwargs
                ):
        remote_u_im, remote_u_tt = remote_u[0], remote_u[1]
        loss1_im, loss1_tt = loss1[0], loss1[1]
        u_im, u_tt = u[0], u[1]
        if dist_student_logit_scale is None:
            dist_student_logit_scale = logit_scale

        results = self.pairwise_loss(remote_features, features, logit_scale.detach(), sim,
                                     ref_features1=ref_remote_features, ref_features2=ref_features)
        diff_im, diff_tt = results["diff_image"], results["diff_text"]

        mask, _ = self.get_mask(diff_im.shape[1], diff_im.shape[0], offset)
        mask = mask.T

        with torch.no_grad():
            weight_im = torch.exp(diff_im - remote_u_im)
            weight_tt = torch.exp(diff_tt - remote_u_tt)
        loss2_im = torch.sum(diff_im * weight_im * mask, dim=-1, keepdim=True) / \
            diff_im.shape[1] * diff_im.shape[0] / (diff_im.shape[0] - 1)
        loss2_tt = torch.sum(diff_tt * weight_tt * mask, dim=-1, keepdim=True) / \
            diff_tt.shape[1] * diff_tt.shape[0] / (diff_tt.shape[0] - 1)

        loss = (torch.mean(loss1_im + loss1_tt) + torch.mean(loss2_im + loss2_tt)) / 2
        if self.multiply_tau:
            loss = loss / logit_scale.detach()
            loss = loss + self.rho / logit_scale
            loss = loss + torch.mean(u_im + u_tt) / 2 / logit_scale
        loss = loss * loss_weight

        if dist_features is not None:
            dist_loss = self.distill_loss(features, dist_features, dist_student_logit_scale, dist_teacher_logit_scale)
            dist_loss = dist_loss * dist_loss_weight
        else:
            dist_loss = torch.tensor(0.0, device=features[0].device)

        if output_dict:
            return {
                "contrastive_loss": loss,
                "distill_loss": dist_loss,
            }
        else:
            return loss, dist_loss


class SogCLRLoss(nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma: float,
                 gamma_schedule: str = "constant",
                 gamma_decay_epochs: int = -1,
                 rho: float = 8.0,
                 eps: float = 1e-14,
                 multiply_tau: bool = True,
                 cache_mask: bool = True,
                 device: torch.device = torch.device("cuda"),
                 loss_weight: float = 1.0,
                 dist_loss_weight: float = 1.0,
                 ssl_loss_weight: float = 1.0,
                 ):
        """Create an instance of Global Contrastive Loss with global temperature parameter."""
        super(SogCLRLoss, self).__init__()
        self.data_size = data_size
        self.gamma = 1.0
        self.gamma_orig = gamma
        self.gamma_schedule = gamma_schedule
        self.gamma_decay_epochs = gamma_decay_epochs
        if self.gamma_schedule != "none":
            assert self.gamma_decay_epochs > 0
        self.rho = rho
        self.eps = eps
        self.multiply_tau = multiply_tau
        self.cache_mask = cache_mask
        self.device = device
        self.loss_weight = loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.ssl_loss_weight = ssl_loss_weight

        self.u_im = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.u_tt = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.arange = {}
        self.mask = {}

        logging.info(f"SogCLR Loss: data size: {data_size}, final gamma: {gamma}, gamma_schedule: {gamma_schedule}, "
                     f"gamma_decay_epochs: {self.gamma_decay_epochs}, rho: {rho}, eps: {self.eps}, "
                     f"multiply_tau: {self.multiply_tau}, cache_mask: {self.cache_mask}, "
                     f"loss_weight: {self.loss_weight}, "
                     f"dist_loss_weight: {self.dist_loss_weight}, ssl_loss_weight: {self.ssl_loss_weight}")

    def adjust_gamma(self, epoch: int):
        if epoch == 0:
            self.gamma = 1.0
        elif epoch >= self.gamma_decay_epochs:
            self.gamma = self.gamma_orig
        else:
            self.gamma = self.gamma_orig
            if self.gamma_schedule == "cosine":
                self.gamma = 0.5 * (1 + math.cos(math.pi * epoch / self.gamma_decay_epochs)) * \
                             (1 - self.gamma_orig) + self.gamma_orig
        logging.info(f"gamma: {self.gamma}")

    def adjust_hyperparams(self, epoch: int):
        self.adjust_gamma(epoch)

    def get_params(self,
                   idx: Optional[Tensor] = None,
                   *args,
                   ):
        results = []
        for src in args:
            results.append(src[idx].to(self.device))
        return results

    def set_params(self,
                   image_idx: Optional[Tensor] = None, text_idx: Optional[Tensor] = None,
                   u_im: Optional[Tensor] = None, u_tt: Optional[Tensor] = None,
                   **kwargs
                   ):
        src_im_list = [u_im]
        dst_im_list = [self.u_im]
        src_tt_list = [u_tt]
        dst_tt_list = [self.u_tt]
        for src_im, dst_im, src_tt, dst_tt in zip(src_im_list, dst_im_list, src_tt_list, dst_tt_list):
            if src_im is not None:
                assert image_idx is not None and dst_im.device == image_idx.device
                dst_im[image_idx] = src_im.to("cpu")
            if src_tt is not None:
                assert text_idx is not None and dst_tt.device == text_idx.device
                dst_tt[text_idx] = src_tt.to("cpu")

    def get_arange(self, length: int, offset: int):
        # here we assume arange is on self.device
        # the arange should be small in size, so we force caching it
        if offset not in self.arange.keys():
            self.arange[offset] = {}
        if length not in self.arange[offset].keys():
            self.arange[offset][length] = torch.arange(length, device=self.device) + offset
        return self.arange[offset][length]

    def get_mask(self, height: int, width: int, offset: int):
        """Return a height * width matrix, with diagonal [offset: offset + height, offset: offset + height]
            being 0 and the rest being 1
        """
        if not self.cache_mask or (height, width, offset) not in self.mask.keys():
            mask_inv = torch.nn.functional.one_hot(self.get_arange(height, offset), width).to(self.device)
            mask = 1 - mask_inv
            if self.cache_mask and (height, width, offset) not in self.mask.keys():
                self.mask[(height, width, offset)] = (mask, mask_inv)
        else:
            mask, mask_inv = self.mask[(height, width, offset)]
        return mask, mask_inv

    def pairwise_loss(self,
                      features1: Tuple[Tensor, Tensor],
                      features2: Tuple[Tensor, Tensor],
                      logit_scale_im: Tensor,
                      sim: Optional[Tuple[Tensor, Tensor]] = None,
                      logit_scale_tt: Optional[Tensor] = None,
                      ref_features1: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_features2: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_sim: Optional[Tuple[Tensor, Tensor]] = None,
                      ):
        image_features1, text_features1 = features1[0], features1[1]
        image_features2, text_features2 = features2[0], features2[1]
        if logit_scale_tt is None:
            logit_scale_tt = logit_scale_im

        if sim is not None:
            sim_image, sim_text = sim[0], sim[1]
        else:
            sim_image = image_features1 @ text_features2.T  # shape [b1, b2]
            sim_text = text_features1 @ image_features2.T  # shape [b1, b2]
        diag_sim = torch.sum(torch.mul(image_features1, text_features1), dim=-1, keepdim=True)

        diff_image = (sim_image - diag_sim).mul(logit_scale_im)
        diff_text = (sim_text - diag_sim).mul(logit_scale_tt)
        if ref_features1 is not None:
            assert ref_features2 is not None
            ref_image_features1, ref_text_features1 = ref_features1[0], ref_features1[1]
            ref_image_features2, ref_text_features2 = ref_features2[0], ref_features2[1]
            if ref_sim is not None:
                ref_sim_image, ref_sim_text = ref_sim[0], ref_sim[1]
            else:
                ref_sim_image = ref_image_features1 @ ref_text_features2.T
                ref_sim_text = ref_text_features1 @ ref_image_features2.T
            ref_diag_sim = torch.sum(torch.mul(ref_image_features1, ref_text_features1), dim=-1, keepdim=True)
            diff_image = diff_image - (ref_sim_image - ref_diag_sim).mul(logit_scale_im)
            diff_text = diff_text - (ref_sim_text - ref_diag_sim).mul(logit_scale_tt)
            diag_sim = diag_sim - ref_diag_sim

        results = {"sim_image": sim_image, "sim_text": sim_text, "diff_image": diff_image,
                   "diff_text": diff_text, "diag_sim": diag_sim}
        return results

    def distill_loss(self,
                     student_features: Tuple[Tensor, Tensor],
                     teacher_features: Tuple[Tensor, Tensor],
                     student_logit_scale: Tensor | float,
                     teacher_logit_scale: Tensor | float | None = None,
                     student_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
                     teacher_remote_features: Optional[Tuple[Tensor, Tensor]] = None,
                     ):
        if teacher_logit_scale is None:
            teacher_logit_scale = student_logit_scale
        if student_remote_features is None:
            if dist.is_initialized():
                student_remote_features = [torch.cat(torch.distributed.nn.all_gather(student_features[0]), dim=0),
                                           torch.cat(torch.distributed.nn.all_gather(student_features[1]), dim=0)]
            else:
                student_remote_features = student_features
        if teacher_remote_features is None:
            if dist.is_initialized():
                teacher_remote_features = [torch.cat(torch.distributed.nn.all_gather(teacher_features[0]), dim=0),
                                           torch.cat(torch.distributed.nn.all_gather(teacher_features[1]), dim=0)]
            else:
                teacher_remote_features = teacher_features

        student_logits_image = student_logit_scale * student_features[0] @ student_remote_features[1].T
        student_logits_text = student_logit_scale * student_features[1] @ student_remote_features[0].T
        teacher_logits_image = teacher_logit_scale * teacher_features[0] @ teacher_remote_features[1].T
        teacher_logits_text = teacher_logit_scale * teacher_features[1] @ teacher_remote_features[0].T

        dist_loss_image = -(teacher_logits_image.softmax(dim=1) * student_logits_image.log_softmax(dim=1)).sum(dim=1).mean()
        dist_loss_text = -(teacher_logits_text.softmax(dim=1) * student_logits_text.log_softmax(dim=1)).sum(dim=1).mean()
        dist_loss = (dist_loss_image + dist_loss_text) / 2

        return dist_loss

    def forward(self,
                features: Tuple[Tensor, Tensor],
                indices: Tuple[Tensor, Tensor],
                logit_scale: Tensor,
                offset: int,
                output_dict: bool = False,
                ref_features: Optional[Tuple[Tensor, Tensor]] = None,
                dist_features: Optional[Tuple[Tensor, Tensor]] = None,
                dist_student_logit_scale: Tensor | None = None,
                dist_teacher_logit_scale: Tensor | float = 100.0,
                **kwargs
                ):
        image_idx, text_idx = indices[0], indices[1]
        u_im = self.get_params(image_idx, self.u_im)[0]
        u_tt = self.get_params(text_idx, self.u_tt)[0]
        if dist_student_logit_scale is None:
            dist_student_logit_scale = logit_scale
        if dist.is_initialized():
            remote_features = (torch.cat(torch.distributed.nn.all_gather(features[0]), dim=0),
                                torch.cat(torch.distributed.nn.all_gather(features[1]), dim=0))
        else:
            remote_features = features
        if ref_features is not None and dist.is_initialized():
            ref_remote_features = (torch.cat(torch.distributed.nn.all_gather(ref_features[0]), dim=0),
                                    torch.cat(torch.distributed.nn.all_gather(ref_features[1]), dim=0))
        else:
            ref_remote_features = ref_features

        results = self.pairwise_loss(features, remote_features, logit_scale,
            ref_features1=ref_features, ref_features2=ref_remote_features)
        diff_im, diff_tt = results["diff_image"], results["diff_text"]
        diag_sim = results["diag_sim"]

        # compute log(eps + g)
        with torch.no_grad():
            mask, mask_inv = self.get_mask(diff_im.shape[0], diff_im.shape[1], offset)
            diff_im_shifted = diff_im + mask_inv * math.log(self.eps * diff_im.shape[1])
            diff_tt_shifted = diff_tt + mask_inv * math.log(self.eps * diff_tt.shape[1])
            logsumexp_im = torch.logsumexp(diff_im_shifted - math.log(self.eps * diff_im.shape[1]),
                                           dim=-1, keepdim=True) + math.log(self.eps)
            logsumexp_tt = torch.logsumexp(diff_tt_shifted - math.log(self.eps * diff_tt.shape[1]),
                                           dim=-1, keepdim=True) + math.log(self.eps)
        if self.gamma == 1.0:
            u_im = logsumexp_im
            u_tt = logsumexp_tt
        else:
            bad_im_idx = torch.nonzero(
                (u_im == 0.0).logical_or(u_im.isinf()).logical_or(u_im.isnan()), as_tuple=True)[0]
            bad_tt_idx = torch.nonzero(
                (u_tt == 0.0).logical_or(u_tt.isinf()).logical_or(u_tt.isnan()), as_tuple=True)[0]
            u_diff_im = u_im + math.log(1- self.gamma) - (logsumexp_im + math.log(self.gamma))
            u_diff_tt = u_tt + math.log(1- self.gamma) - (logsumexp_tt + math.log(self.gamma))
            u_im = torch.where(u_diff_im > 0,
                               u_im + math.log(1- self.gamma) + torch.log(1 + torch.exp(-u_diff_im)),
                               logsumexp_im + math.log(self.gamma) + torch.log(1 + torch.exp(u_diff_im)))
            u_tt = torch.where(u_diff_tt > 0,
                               u_tt + math.log(1- self.gamma) + torch.log(1 + torch.exp(-u_diff_tt)),
                               logsumexp_tt + math.log(self.gamma) + torch.log(1 + torch.exp(u_diff_tt)))
            if bad_im_idx.shape[0] > 0:
                u_im[bad_im_idx] = logsumexp_im[bad_im_idx].to(u_im.dtype)
            if bad_tt_idx.shape[0] > 0:
                u_tt[bad_tt_idx] = logsumexp_tt[bad_tt_idx].to(u_tt.dtype)

        with torch.no_grad():
            weight_im = torch.exp(diff_im - u_im)
            weight_tt = torch.exp(diff_tt - u_tt)
        loss_im = torch.sum(diff_im * weight_im * mask, dim=-1, keepdim=True) / (diff_im.shape[1] - 1)
        loss_tt = torch.sum(diff_tt * weight_tt * mask, dim=-1, keepdim=True) / (diff_tt.shape[1] - 1)

        gather_list = [u_im, u_tt]

        loss = torch.mean(loss_im + loss_tt)
        if self.multiply_tau:
            loss = loss / logit_scale.detach()
            loss = loss + self.rho / logit_scale
            loss = loss + torch.mean(u_im + u_tt) / 2 / logit_scale
        loss = loss * self.loss_weight
        loss_dict = {"contrastive_loss": loss, "gather_list": gather_list}

        if dist_features is not None:
            dist_loss = self.distill_loss(features, dist_features, dist_student_logit_scale, dist_teacher_logit_scale)
            dist_loss = dist_loss * self.dist_loss_weight
            if self.multiply_tau:
                dist_loss = dist_loss / logit_scale.detach()
            loss_dict["distill_loss"] = dist_loss
        else:
            dist_loss = torch.tensor(0.0, device=features[0].device)

        if output_dict:
            return loss_dict
        else:
            return loss, dist_loss, gather_list


class FastCLIPLossIndividual(FastCLIPLoss):
    def __init__(self,
                 data_size: int,
                 tau_init: float,
                 lr_tau: float,
                 beta1_tau: float = 0.9,
                 beta2_tau: float = 0.999,
                 eps_tau: float = 1e-8,
                 device: torch.device = torch.device("cuda"),
                 **kwargs
                 ):
        """Create an instance of Global Contrastive Loss with individual temperature parameters.
            This is a subclass of FastCLIPLoss, with additional parameters for individual temperature parameters.
        """
        super().__init__(data_size=data_size, device=device, **kwargs)
        self.tau_im = torch.ones(data_size, device="cpu").reshape(-1, 1) * tau_init
        self.tau_tt = torch.ones(data_size, device="cpu").reshape(-1, 1) * tau_init

        self.beta1_tau_orig = beta1_tau
        self.beta1_tau = 0.0
        self.beta2_tau_orig = beta2_tau
        self.beta2_tau = 0.0
        self.grad_clamp_tau = 5.0
        self.eps_tau = eps_tau
        self.epoch = 0

        self.m_grad_tau_im = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.m_grad_tau_tt = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.v_grad_tau_im = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.v_grad_tau_tt = torch.zeros(data_size, device="cpu").reshape(-1, 1)

        self.tau_min, self.tau_max = 0.01, 1.0
        self.lr_tau = lr_tau

        self.bound_im = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.bound_tt = torch.zeros(data_size, device="cpu").reshape(-1, 1)

        logging.info(f"beta1_tau: {self.beta1_tau_orig}, beta2_tau: {self.beta2_tau_orig}, eps_tau: {self.eps_tau}")

    def adjust_hyperparams(self, epoch: int):
        self.epoch = epoch
        self.adjust_gamma(epoch)
        # self.update_lr_tau(epoch)
        if epoch > 0:
            self.beta1_tau = self.beta1_tau_orig
            self.beta2_tau = self.beta2_tau_orig

    def set_params(self,
                   image_idx: Optional[Tensor] = None, text_idx: Optional[Tensor] = None,
                   u_im: Optional[Tensor] = None, u_tt: Optional[Tensor] = None,
                   tau_im: Optional[Tensor] = None, tau_tt: Optional[Tensor] = None,
                   bound_im: Optional[Tensor] = None, bound_tt: Optional[Tensor] = None,
                   m_grad_tau_im: Optional[Tensor] = None, m_grad_tau_tt: Optional[Tensor] = None,
                   v_grad_tau_im: Optional[Tensor] = None, v_grad_tau_tt: Optional[Tensor] = None,
                   **kwargs
                   ):
        src_im_list = [u_im, tau_im, bound_im, m_grad_tau_im, v_grad_tau_im]
        dst_im_list = [self.u_im, self.tau_im, self.bound_im, self.m_grad_tau_im, self.v_grad_tau_im]
        src_tt_list = [u_tt, tau_tt, bound_tt, m_grad_tau_tt, v_grad_tau_tt]
        dst_tt_list = [self.u_tt, self.tau_tt, self.bound_tt, self.m_grad_tau_tt, self.v_grad_tau_tt]
        for src_im, dst_im, src_tt, dst_tt in zip(src_im_list, dst_im_list, src_tt_list, dst_tt_list):
            if src_im is not None:
                assert image_idx is not None and dst_im.device == image_idx.device
                dst_im[image_idx] = src_im.to("cpu")
            if src_tt is not None:
                assert text_idx is not None and dst_tt.device == text_idx.device
                dst_tt[text_idx] = src_tt.to("cpu")

    def pairwise_loss(self,
                      features1: Tuple[Tensor, Tensor],
                      features2: Tuple[Tensor, Tensor],
                      logit_scale_im: Tensor,
                      offset: int = 0,
                      sim: Optional[Tuple[Tensor, Tensor]] = None,
                      logit_scale_tt: Optional[Tensor] = None,
                      bounds: Optional[Tuple[Tensor, Tensor]] = None,
                      update_bounds: bool = True,
                      ref_features1: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_features2: Optional[Tuple[Tensor, Tensor]] = None,
                      ref_sim: Optional[Tuple[Tensor, Tensor]] = None,
                      ):
        # TODO implement stabilized u update for FastCLIPLossIndividual
        image_features1, text_features1 = features1[0], features1[1]
        image_features2, text_features2 = features2[0], features2[1]
        if logit_scale_tt is None:
            logit_scale_tt = logit_scale_im

        batch_size1 = image_features1.shape[0]  # b1
        batch_size2 = image_features2.shape[0]  # b2

        if sim is not None:
            sim_image, sim_text = sim[0], sim[1]
        else:
            sim_image = image_features1 @ text_features2.T  # shape [b1, b2]
            sim_text = text_features1 @ image_features2.T  # shape [b1, b2]
        diag_sim = torch.sum(torch.mul(image_features1, text_features1), dim=-1, keepdim=True)

        diff_image = (sim_image - diag_sim).mul(logit_scale_im)
        diff_text = (sim_text - diag_sim).mul(logit_scale_tt)
        if ref_features1 is not None:
            assert ref_features2 is not None
            ref_image_features1, ref_text_features1 = ref_features1[0], ref_features1[1]
            ref_image_features2, ref_text_features2 = ref_features2[0], ref_features2[1]
            if ref_sim is not None:
                ref_sim_image, ref_sim_text = ref_sim[0], ref_sim[1]
            else:
                ref_sim_image = ref_image_features1 @ ref_text_features2.T
                ref_sim_text = ref_text_features1 @ ref_image_features2.T
            ref_diag_sim = torch.sum(torch.mul(ref_image_features1, ref_text_features1), dim=-1, keepdim=True)
            diff_image = diff_image - (ref_sim_image - ref_diag_sim).mul(logit_scale_im)
            diff_text = diff_text - (ref_sim_text - ref_diag_sim).mul(logit_scale_tt)

        bounds_image, bounds_text = None, None
        if bounds is not None:
            bounds_image, bounds_text = bounds[0], bounds[1]
            if update_bounds:
                bounds_image = torch.maximum(
                    bounds_image, torch.max(diff_image, dim=-1, keepdim=True).values.detach())
                bounds_text = torch.maximum(
                    bounds_text, torch.max(diff_text, dim=-1, keepdim=True).values.detach())
            diff_image = diff_image.sub(bounds_image)
            diff_text = diff_text.sub(bounds_text)
        exp_diff_image = torch.exp(diff_image)
        exp_diff_text = torch.exp(diff_text)

        if batch_size1 <= batch_size2:
            mask, mask_inv = self.get_mask(batch_size1, batch_size2, offset)
        else:
            mask, mask_inv = self.get_mask(batch_size2, batch_size1, offset)
            mask, mask_inv = mask.T, mask_inv.T
        exp_diff_image = torch.mul(exp_diff_image, mask)
        exp_diff_text = torch.mul(exp_diff_text, mask)

        if batch_size1 <= batch_size2:
            real_weights_sum = batch_size2 - 1
        else:
            real_weights_sum = batch_size2 * (batch_size1 - 1) / batch_size1
        loss_image = torch.sum(exp_diff_image, dim=-1, keepdim=True) / real_weights_sum
        loss_text = torch.sum(exp_diff_text, dim=-1, keepdim=True) / real_weights_sum

        results = {"loss_image": loss_image, "loss_text": loss_text, "sim_image": sim_image, "sim_text": sim_text,
                   "exp_diff_image": exp_diff_image, "exp_diff_text": exp_diff_text, "diff_image": diff_image,
                   "diff_text": diff_text, "bounds_image": bounds_image, "bounds_text": bounds_text,
                   "mask": mask}
        return results

    def local(self,
              features: Tuple[Tensor, Tensor],
              indices: Tuple[Tensor, Tensor],
              remote_features: Tuple[Tensor, Tensor],
              offset: int,
              **kwargs
              ):
        image_idx, text_idx = indices[0], indices[1]
        u_im, tau_im, bound_im, m_grad_tau_im, v_grad_tau_im = self.get_params(
            image_idx, self.u_im, self.tau_im, self.bound_im, self.m_grad_tau_im, self.v_grad_tau_im)
        u_tt, tau_tt, bound_tt, m_grad_tau_tt, v_grad_tau_tt = self.get_params(
            text_idx, self.u_tt, self.tau_tt, self.bound_tt, self.m_grad_tau_tt, self.v_grad_tau_tt)
        bounds = (bound_im.clone(), bound_tt.clone())

        results = self.pairwise_loss(features, remote_features, 1.0/tau_im, offset=offset,
                                     logit_scale_tt=1.0/tau_tt, bounds=bounds)
        loss1_im, loss1_tt = results["loss_image"], results["loss_text"]
        sim_im, sim_tt = results["sim_image"], results["sim_text"]
        exp_diff_im, exp_diff_tt = results["exp_diff_image"], results["exp_diff_text"]
        diff_im, diff_tt = results["diff_image"], results["diff_text"]
        bound_im, bound_tt = results["bounds_image"], results["bounds_text"]
        assert bound_im is not None and bound_tt is not None

        g_im = loss1_im.detach()
        g_tt = loss1_tt.detach()
        if self.gamma < 1.0:
            bad_im_idx = torch.nonzero(
                (u_im < 1e-35).logical_or(u_im.isinf()).logical_or(u_im.isnan()), as_tuple=True)[0]
            bad_tt_idx = torch.nonzero(
                (u_tt < 1e-35).logical_or(u_tt.isinf()).logical_or(u_tt.isnan()), as_tuple=True)[0]
        u_im = (1.0 - self.gamma) * u_im * torch.exp(bounds[0] - bound_im) + self.gamma * g_im
        u_tt = (1.0 - self.gamma) * u_tt * torch.exp(bounds[1] - bound_tt) + self.gamma * g_tt
        if self.gamma < 1.0:
            if bad_im_idx.shape[0] > 0:
                u_im[bad_im_idx] = g_im[bad_im_idx].to(u_im.dtype)
            if bad_tt_idx.shape[0] > 0:
                u_tt[bad_tt_idx] = g_tt[bad_tt_idx].to(u_tt.dtype)

        batch_size = remote_features[0].shape[0] - 1
        # note that here diff is subtracted by new_bounds in pairwise_loss()
        # here we do not divide the gradient by dataset size,
        # since it is equivalent to dividing lr_tau by dataset size
        grad_tau_im = (-1 * exp_diff_im.mul(diff_im.add(bound_im)).sum(dim=-1, keepdim=True).div(u_im + self.eps).div(batch_size)
                       + torch.log(u_im + self.eps) + bound_im + self.rho).detach().clamp_(min=-self.grad_clamp_tau, max=self.grad_clamp_tau)
        grad_tau_tt = (-1 * exp_diff_tt.mul(diff_tt.add(bound_tt)).sum(dim=-1, keepdim=True).div(u_tt + self.eps).div(batch_size)
                       + torch.log(u_tt + self.eps) + bound_tt + self.rho).detach().clamp_(min=-self.grad_clamp_tau, max=self.grad_clamp_tau)

        m_grad_tau_im = self.beta1_tau * m_grad_tau_im + (1.0 - self.beta1_tau) * grad_tau_im
        m_grad_tau_tt = self.beta1_tau * m_grad_tau_tt + (1.0 - self.beta1_tau) * grad_tau_tt
        v_grad_tau_im = self.beta2_tau * v_grad_tau_im + (1.0 - self.beta2_tau) * grad_tau_im ** 2
        v_grad_tau_tt = self.beta2_tau * v_grad_tau_tt + (1.0 - self.beta2_tau) * grad_tau_tt ** 2
        m_hat_grad_tau_im = m_grad_tau_im / (1.0 - self.beta1_tau ** (self.epoch + 1))
        m_hat_grad_tau_tt = m_grad_tau_tt / (1.0 - self.beta1_tau ** (self.epoch + 1))
        v_hat_grad_tau_im = v_grad_tau_im / (1.0 - self.beta2_tau ** (self.epoch + 1))
        v_hat_grad_tau_tt = v_grad_tau_tt / (1.0 - self.beta2_tau ** (self.epoch + 1))

        tau_im = (tau_im - self.lr_tau * m_hat_grad_tau_im / (v_hat_grad_tau_im + self.eps_tau)).clamp_(min=self.tau_min, max=self.tau_max)
        tau_tt = (tau_tt - self.lr_tau * m_hat_grad_tau_tt / (v_hat_grad_tau_tt + self.eps_tau)).clamp_(min=self.tau_min, max=self.tau_max)

        gather_list = [u_im, u_tt, tau_im, tau_tt, bound_im, bound_tt,
                       m_grad_tau_im, m_grad_tau_tt, v_grad_tau_im, v_grad_tau_tt]

        return loss1_im, loss1_tt, sim_im, sim_tt, gather_list

    def forward(self,
                features: Tuple[Tensor, Tensor],
                remote_features: Tuple[Tensor, Tensor],
                remote_u: Tuple[Tensor, Tensor],
                remote_tau: Tuple[Tensor, Tensor],
                remote_bounds: Tuple[Tensor, Tensor],
                loss1: Tuple[Tensor, Tensor],
                u: Tuple[Tensor, Tensor],
                sim: Tuple[Tensor, Tensor],
                offset: int,
                output_dict: bool = False,
                **kwargs
                ):
        remote_u_im, remote_u_tt = remote_u[0], remote_u[1]
        loss1_im, loss1_tt = loss1[0], loss1[1]
        u_im, u_tt = u[0], u[1]
        remote_tau_im, remote_tau_tt = remote_tau[0], remote_tau[1]

        results = self.pairwise_loss(remote_features, features, 1.0/remote_tau_im, offset=offset, sim=sim,
                                     logit_scale_tt=1.0/remote_tau_tt, bounds=remote_bounds, update_bounds=False)
        loss2_im, loss2_tt = results["loss_image"], results["loss_text"]

        partial_grad1_im = loss1_im / (u_im + self.eps)
        partial_grad1_tt = loss1_tt / (u_tt + self.eps)
        partial_grad2_im = loss2_im / (remote_u_im + self.eps)
        partial_grad2_tt = loss2_tt / (remote_u_tt + self.eps)
        loss = (torch.mean(partial_grad1_im + partial_grad1_tt)
                + torch.mean(partial_grad2_im + partial_grad2_tt)) / 2

        if output_dict:
            return {
                "contrastive_loss": loss,
            }
        else:
            return loss
