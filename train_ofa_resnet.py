# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#
# Training script for OFA-ResNet50 supernet.
# Launch with torchrun:
#   Single-GPU:   python train_ofa_resnet.py --task expand --phase 1
#   Multi-GPU:    torchrun --nproc_per_node=8 train_ofa_resnet.py --task expand --phase 1

import argparse
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.utils import MyRandomResizedCrop
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    load_models,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="expand",
    choices=["expand", "width", "depth"],
    help=(
        "Progressive-shrinking stage. "
        "Train in order: expand → width → depth."
    ),
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "--imagenet_path",
    type=str,
    default=None,
    help="Path to ImageNet dataset root (contains train/ and val/).",
)
parser.add_argument(
    "--ofa_checkpoint_path",
    type=str,
    default=None,
    help=(
        "Checkpoint to load at the start of this stage. "
        "Provide a pretrained ResNet50D checkpoint for the first expand phase, "
        "or the previous stage checkpoint for later phases."
    ),
)
parser.add_argument("--kd_ratio", type=float, default=1.0)
parser.add_argument("--kd_type", type=str, default="ce", choices=["ce", "mse"])
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument(
    "--base_checkpoint_dir",
    type=str,
    default="/coc/scratch/dgarg/ofa_checkpoints",
    help="Root directory for all checkpoint subdirectories.",
)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Stage-specific hyper-parameters
# OFA-ResNet50 design space:
#   depth_list  = [0, 1, 2]  (extra blocks per stage on top of base depths)
#   expand_list = [0.2, 0.25, 0.35]
#   width_list  = [0.65, 0.8, 1.0]
# ---------------------------------------------------------------------------
if args.task == "expand":
    # Stage 1: add elastic expand-ratio (depth and width are at max)
    args.path = os.path.join(args.base_checkpoint_dir, "expand", "phase%d" % args.phase)
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = "0.25,0.35"
        args.depth_list = "2"
        args.width_mult_list = "1.0"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = "0.2,0.25,0.35"
        args.depth_list = "2"
        args.width_mult_list = "1.0"

elif args.task == "width":
    # Stage 2: add elastic width multiplier
    args.path = os.path.join(args.base_checkpoint_dir, "width", "phase%d" % args.phase)
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = "0.2,0.25,0.35"
        args.depth_list = "2"
        args.width_mult_list = "0.8,1.0"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = "0.2,0.25,0.35"
        args.depth_list = "2"
        args.width_mult_list = "0.65,0.8,1.0"

elif args.task == "depth":
    # Stage 3: add elastic depth
    args.path = os.path.join(args.base_checkpoint_dir, "depth", "phase%d" % args.phase)
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = "0.2,0.25,0.35"
        args.depth_list = "1,2"
        args.width_mult_list = "0.65,0.8,1.0"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = "0.2,0.25,0.35"
        args.depth_list = "0,1,2"
        args.width_mult_list = "0.65,0.8,1.0"
else:
    raise NotImplementedError

args.manual_seed = 0
args.lr_schedule_type = "cosine"
args.base_batch_size = 64
args.valid_size = 10000
args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False
args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10
args.resize_scale = 0.08
args.distort_color = "tf"
args.image_size = "128,160,192,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False
args.teacher_model = None  # ResNet50 uses no KD teacher by default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    # Distributed setup (works with torchrun and single-process)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        is_root = dist.get_rank() == 0
        num_gpus = dist.get_world_size()
    else:
        local_rank = 0
        is_root = True
        num_gpus = 1
        torch.cuda.set_device(0)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # Image size
    args.image_size = [int(s) for s in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # Run config
    args.lr_schedule_param = None
    args.opt_param = {"momentum": args.momentum, "nesterov": not args.no_nesterov}
    args.init_lr = args.base_lr * num_gpus
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4

    if args.imagenet_path is not None:
        os.environ.setdefault("IMAGENET_PATH", args.imagenet_path)
        from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
        ImagenetDataProvider.DEFAULT_PATH = args.imagenet_path

    run_config = DistributedImageNetRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=(dist.get_rank() if dist.is_initialized() else 0)
    )

    if is_root:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # Build OFA-ResNet50 supernet
    args.width_mult_list = [float(w) for w in args.width_mult_list.split(",")]
    args.expand_list = [float(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    net = OFAResNets(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        depth_list=args.depth_list,
        expand_ratio_list=args.expand_list,
        width_mult_list=args.width_mult_list,
    )

    # Distributed run manager
    run_manager = DistributedRunManager(
        args.path,
        net,
        run_config,
        backward_steps=args.dynamic_batch_size,
        is_root=is_root,
    )
    run_manager.save_config()
    if args.resume:
        run_manager.load_model()  # restores weights, optimizer state, and start_epoch from checkpoint.pth.tar
    run_manager.broadcast()  # syncs start_epoch and weights from rank-0 to all workers

    # Training
    from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
        validate,
        train,
        train_elastic_depth,
        train_elastic_expand,
        train_elastic_width_mult,
    )

    validate_func_dict = {
        "image_size_list": {224}
        if isinstance(args.image_size, int)
        else sorted({160, 224}),
        "ks_list": [3],  # ResNet50 has fixed kernel size 3
        "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        "width_mult_list": sorted({0, len(net.width_mult_list) - 1}),
    }

    if args.task == "expand":
        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
            train_elastic_expand,
        )

        if run_manager.start_epoch == 0 and not args.resume:
            if args.ofa_checkpoint_path is None:
                if is_root:
                    print(
                        "WARNING: No --ofa_checkpoint_path provided for expand phase 1. "
                        "Starting from random init. For best results, provide a pretrained "
                        "ResNet50D checkpoint (max expand=0.35, width=1.0, depth=2)."
                    )
            else:
                load_models(run_manager, net, model_path=args.ofa_checkpoint_path)
                run_manager.write_log(
                    "%.3f\t%.3f\t%.3f\t%s"
                    % validate(run_manager, is_test=True, **validate_func_dict),
                    "valid",
                )
        else:
            assert args.resume
        train_elastic_expand(train, run_manager, args, validate_func_dict)

    elif args.task == "width":
        if args.ofa_checkpoint_path is None:
            raise ValueError(
                "--ofa_checkpoint_path is required for width stage. "
                "Provide the checkpoint from the completed expand phase 2."
            )
        train_elastic_width_mult(train, run_manager, args, validate_func_dict)

    elif args.task == "depth":
        if args.ofa_checkpoint_path is None:
            raise ValueError(
                "--ofa_checkpoint_path is required for depth stage. "
                "Provide the checkpoint from the completed width phase 2."
            )
        train_elastic_depth(train, run_manager, args, validate_func_dict)

    else:
        raise NotImplementedError

    if dist.is_initialized():
        dist.destroy_process_group()
