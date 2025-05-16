import argparse
import jsonargparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    # parser = argparse.ArgumentParser()
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use, can be ['adamw', 'lamb'].")
    parser.add_argument("--lr_tau", type=float, default=-1.0,
                        help="Learning rate of the temperature parameter. If < 0, will be set to lr of the model.")
    parser.add_argument("--lr_tau_scheduler", type=str, default="cosine", help="Learning rate scheduler for tau.")
    parser.add_argument("--temperature_scheme", type=str, default="global_learnable",
                        help=("Temperature scheme for FastCLIP. Combination of"
                              " ['global', 'individual'] (only works for FastCLIP) and ['learnable', 'constant']."))
    parser.add_argument("--profile", default=False, action="store_true", help="Whether to profile training stats.")
    parser.add_argument("--config_file", action=jsonargparse.ActionConfigFile,
                        help="Optional configuration file in JSON or YAML format.")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs to use.")
    parser.add_argument("--dist_port", type=str, default="29500", help="Port for distributed training.")
    parser.add_argument("--stop_epochs", type=int, default=-1,
                        help=("The training will stop after this epoch. Useful when the training needs"
                              " to be split into several parts. If < 0, will stop after --epochs"))
    parser.add_argument("--num_samples_per_shard", type=int, default=10000,
                        help=("Number of samples per shard in webdataset. Used when caching and sharding"
                              " reference model features."))
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer.")
    parser.add_argument("--iters", type=int, default=-1,
                        help=("Total number of iterations."
                              " If given, the learning rate schedule will be based on this value."))
    parser.add_argument("--stop_iters", type=int, default=-1, help="Stop training after this many iterations.")
    parser.add_argument("--logit_scale_bound", type=float, default=100.0, help="Upper bound for logit scale.")
    parser.add_argument("--lr_min", type=float, default=0.0,
                        help=("The minimum learning rate of the model and the temperature parameter."
                              " Only works when using the cosine scheduler."))

    # fastclip
    parser.add_argument("--fastclip", default=False, action="store_true", help="Whether to use FastCLIP losses.")
    parser.add_argument("--gamma", type=float, default=0.8, help="Inner learning rate in FastCLIP.")
    parser.add_argument("--gamma_schedule", type=str, default="cosine",
                        help="Schedule for gamma in FastCLIP, can be ['constant', 'cosine'].")
    parser.add_argument("--gamma_decay_epochs", type=int, default=10, help="Decay epochs for gamma in FastCLIP.")
    parser.add_argument("--rho", type=float, default=8.0, help="Rho in FastCLIP.")
    parser.add_argument("--fastclip_eps", type=float, default=1e-14, help="Epsilon in FastCLIP.")
    parser.add_argument("--multiply_tau", default=False, action="store_true",
                        help="Whether to multiply the FastCLIP loss by tau.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Initial temperature value.")
    parser.add_argument("--data_size", type=int, default=-1, help="Original dataset size.")
    parser.add_argument("--ref_model", type=str, default="", help="Reference model to compute the reference loss")
    parser.add_argument("--ref_model_pretrained", type=str, default="",
                        help="Pretrained checkpoint name for the reference model")
    parser.add_argument("--ref_model_checkpoint", type=str, default="",
                        help="Path to load the reference model. Only used if --ref_model_pretrained is not set.")
    parser.add_argument("--cache_ref_model_features", default=False, action="store_true",
                        help=("Whether to cache the features of the reference model."
                              " If True, training will not be done, only the reference features will be cached."))
    parser.add_argument("--cached_ref_features_dir", type=str, default="",
                        help="Directory to store the cached features of the reference model.")
    parser.add_argument("--ref_features_offset", type=int, default=0,
                        help="Offset to cache reference model features.")
    parser.add_argument("--ref_features_usage", type=str, default="ref",
                        help="How to use the reference features, can be combination of ['ref', 'distill', 'select'].")
    parser.add_argument("--ref_filter_ratio", type=float, default=0.8,
                        help="Ratio of discarded examples in example selection.")
    parser.add_argument("--ref_filter_n_chunks", type=int, default=2,
                        help="Nubmer of chunks to use in example selection.")
    parser.add_argument("--ref_filter_topk", default=False, action="store_true",
                        help="Whether to use topk selection or sampleing in example selection.")
    parser.add_argument("--distill_weight", type=float, default=1.0,
                        help="Weight for the distillation loss.")
    parser.add_argument("--distill_ref_logit_scale", type=float, default=0.0,
                        help="Logit scale used in reference loss when ref_features_usage is 'distill_ref'")
    parser.add_argument("--distill_logit_scale", type=float, default=100.0,
                        help="Logit scale of the teacher in distillation loss")
    parser.add_argument("--distill_teacher_dimension", type=int, default=[-1], nargs="+",
                        help="Number of dimensions for each teacher. Default: [-1].")
    parser.add_argument("--distill_average_after_softmax", default=False, action="store_true",
                        help='For ensemble models in distillation, average logits after Softmax.')
    parser.add_argument("--cap_model", type=str, default="", help="Captioning model to generate synthetic captions.")
    parser.add_argument("--cap_model_pretrained", type=str, default="",
                        help="Pretrained checkpoint name for the captioning model")
    parser.add_argument("--rand_augment", default=False, action="store_true",
                        help="Whether to apply RandAugment to images.")
    parser.add_argument("--cache_syn_texts", default=False, action="store_true",
                        help="Whether to cache the synthetic captions generated by the captioning model.")
    parser.add_argument("--cached_syn_texts_dir", type=str, default="",
                        help="Directory to store the cached synthetic captions generated by the captioning model.")
    parser.add_argument("--syn_texts_offset", type=int, default=0,
                        help="Offset to generate synthetic captions.")
    parser.add_argument("--num_syn_texts", type=int, default=5,
                        help="Number of generated synthetic captions for each sample.")
    parser.add_argument("--skip_webdataset_split_by_node", default=False, action="store_true",
                        help="Whether to skip splitting the webdataset by node.")
    parser.add_argument("--cache_group_size", type=int, default=1,
                        help="Group size for caching reference model features or synthetic texts.")
    parser.add_argument("--cache_rank", type=int, default=0,
                        help="Rank for caching reference model features or synthetic texts.")
    parser.add_argument("--loss_weight", type=float, default=1.0,
                        help="Weight for the contrastive loss.")

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
