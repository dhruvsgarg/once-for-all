# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#
# Training script for OFA-ResNet50 supernet.
# Launch with torchrun:
#   Single-GPU:   python train_ofa_resnet.py --task expand --phase 1
#   Multi-GPU:    torchrun --nproc_per_node=8 train_ofa_resnet.py --task expand --phase 1
#
# Individual subnet fine-tuning (6 specific subnets from a JSON config):
#   --ofa_checkpoint_path is the INPUT base ResNet50D checkpoint (read-only).
#   --subnet_out_dir is where each fine-tuned subnet is written (default:
#     /coc/scratch/dgarg/ofa_checkpoints/finetuned_subnets, separate from OFA stage dirs).
#
#   python train_ofa_resnet.py --train_subnets \
#       --subnet_config_json latency_curves_supernet_resnet_A40_with_stages_29apr26.json \
#       --ofa_checkpoint_path /coc/scratch/dgarg/resnet50d_base.pth.tar \
#       --subnet_out_dir /coc/scratch/dgarg/ofa_checkpoints/finetuned_subnets \
#       --subnet_epochs 30 --subnet_lr 2.5e-3

import argparse
import json
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
from ofa.utils import MyRandomResizedCrop, list_mean
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
        "INPUT checkpoint to load weights from (never written to). "
        "For --train_subnets: path to the pretrained ResNet50D base checkpoint "
        "(.pth.tar with a 'state_dict' key). "
        "For progressive shrinking: path to the completed checkpoint from the "
        "previous stage (expand → width → depth)."
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

# ---------------------------------------------------------------------------
# Subnet-training flags
# ---------------------------------------------------------------------------
parser.add_argument(
    "--train_subnets",
    action="store_true",
    help=(
        "Instead of full progressive-shrinking supernet training, fine-tune "
        "each specific subnet defined in --subnet_config_json. "
        "Requires --ofa_checkpoint_path (pretrained OFA supernet or ResNet50D "
        "checkpoint) and --subnet_config_json."
    ),
)
parser.add_argument(
    "--subnet_config_json",
    type=str,
    default=None,
    help=(
        "Path to JSON file containing the 6 subnet configurations to train. "
        "Each entry must have subnet_dimension.depth_values, "
        "subnet_dimension.elasticity_ratio, subnet_dimension.width_multiplier, "
        "and accuracy (expected top-1 %%). Required when --train_subnets is set."
    ),
)
parser.add_argument(
    "--subnet_out_dir",
    type=str,
    default=None,
    help=(
        "Output root for subnet fine-tuning checkpoints. "
        "Each subnet is saved under <subnet_out_dir>/subnet_<id>/. "
        "Defaults to /coc/scratch/dgarg/ofa_checkpoints/finetuned_subnets so it stays "
        "separate from the OFA progressive-shrinking stage directories."
    ),
)
parser.add_argument(
    "--subnet_epochs",
    type=int,
    default=100,
    help="Fine-tuning epochs per subnet when --train_subnets is set.",
)
parser.add_argument(
    "--subnet_lr",
    type=float,
    default=2.5e-3,
    help="Base learning rate per GPU for subnet fine-tuning.",
)
parser.add_argument(
    "--subnet_ids",
    type=str,
    default=None,
    help=(
        "Optional comma-separated subnet ids to train (e.g., '0,2,5'). "
        "If omitted, all subnets in --subnet_config_json are trained."
    ),
)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Stage-specific hyper-parameters
# OFA-ResNet50 design space:
#   depth_list  = [0, 1, 2]  (extra blocks per stage on top of base depths)
#   expand_list = [0.2, 0.25, 0.35]
#   width_list  = [0.65, 0.8, 1.0]
# ---------------------------------------------------------------------------
if args.train_subnets:
    # Subnet training bypasses the progressive-shrinking stage setup.
    # We set stubs so downstream code that unconditionally reads these attributes
    # (e.g. image-size parsing) still finds valid values.
    if args.subnet_out_dir is None:
        args.subnet_out_dir = "/coc/scratch/dgarg/ofa_checkpoints/finetuned_subnets"
    args.path = args.subnet_out_dir  # used only for os.makedirs in __main__
    args.dynamic_batch_size = 1
    args.n_epochs = args.subnet_epochs
    args.base_lr = args.subnet_lr
    args.warmup_epochs = 0
    args.warmup_lr = args.subnet_lr
    args.expand_list = "0.2,0.25,0.35"
    args.depth_list = "0,1,2"
    args.width_mult_list = "0.65,0.8,1.0"

elif args.task == "expand":
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
# Subnet fine-tuning
# ---------------------------------------------------------------------------

def train_individual_subnets(args, run_config, is_root, num_gpus):
    """
    Fine-tune each subnet from args.subnet_config_json.

    Weights are initialised from the OFA supernet checkpoint at
    args.ofa_checkpoint_path, then the fixed-architecture subnet is
    extracted and fine-tuned as a standalone model.  Each subnet is
    saved independently so it can be loaded directly at serving time
    without any OFA infrastructure.

    Expected accuracy numbers come from the JSON file and are printed
    alongside achieved accuracy at the end so you can verify the
    training was successful before deploying.
    """
    if args.subnet_config_json is None:
        raise ValueError(
            "--subnet_config_json is required when --train_subnets is set."
        )
    if args.ofa_checkpoint_path is None:
        raise ValueError(
            "--ofa_checkpoint_path is required when --train_subnets is set. "
            "Provide a pretrained OFA supernet checkpoint (or a ResNet50D "
            "checkpoint that covers the full design space)."
        )

    with open(args.subnet_config_json) as f:
        config_data = json.load(f)
    subnets_cfg = config_data["models"]
    selected_ids = None
    if args.subnet_ids:
        selected_ids = {int(x) for x in args.subnet_ids.split(",") if x.strip()}
        subnets_cfg = [s for s in subnets_cfg if s.get("id") in selected_ids]
        if is_root:
            print(f"Filtering subnets to ids: {sorted(selected_ids)}")

    if is_root:
        print(
            f"\n{'='*70}\n"
            f"Subnet fine-tuning mode: {len(subnets_cfg)} subnets\n"
            f"Checkpoint : {args.ofa_checkpoint_path}\n"
            f"Epochs/subnet: {args.subnet_epochs}  |  "
            f"LR: {args.subnet_lr} x {num_gpus} GPUs = "
            f"{args.subnet_lr * num_gpus:.4f}\n"
            f"{'='*70}"
        )

    # Build OFA supernet with the full design space so we can extract any
    # of the 6 subnets regardless of which config they came from.
    full_net = OFAResNets(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        depth_list=[0, 1, 2],
        expand_ratio_list=[0.2, 0.25, 0.35],
        width_mult_list=[0.65, 0.8, 1.0],
    )
    full_net.cuda()

    # Load pretrained weights from the provided checkpoint.
    # OFAResNets.load_state_dict() does its own key remapping (bn. → bn.bn.,
    # conv.weight → conv.conv.weight, etc.) and returns None, so we cannot
    # unpack (missing, unexpected) from it.
    ckpt = torch.load(args.ofa_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    full_net.load_state_dict(state_dict)
    if is_root:
        print(f"  Loaded supernet weights from {args.ofa_checkpoint_path}")

    # Broadcast loaded weights from rank 0 to all ranks.
    if dist.is_initialized():
        for param in full_net.parameters():
            dist.broadcast(param.data, src=0)
        for buf in full_net.buffers():
            dist.broadcast(buf, src=0)

    # Subnet-specific training args (shared across all subnets).
    subnet_args = argparse.Namespace(**vars(args))
    subnet_args.n_epochs = args.subnet_epochs
    subnet_args.base_lr = args.subnet_lr
    subnet_args.warmup_epochs = 0
    subnet_args.warmup_lr = args.subnet_lr
    subnet_args.teacher_model = None
    subnet_args.kd_ratio = 0.0

    # Patch run_config so DistributedRunManager's LR scheduler uses the
    # right epoch count and initial LR.
    run_config.n_epochs = args.subnet_epochs
    run_config.init_lr = args.subnet_lr * num_gpus

    results = []

    for subnet_info in subnets_cfg:
        subnet_id = subnet_info["id"]
        expected_acc = subnet_info["accuracy"]
        dim = subnet_info["subnet_dimension"]
        depth_values = dim["depth_values"]
        elasticity_ratio = dim["elasticity_ratio"]
        width_multiplier = dim["width_multiplier"]

        if is_root:
            print(
                f"\n{'-'*70}\n"
                f"Subnet {subnet_id}  |  expected top-1: {expected_acc:.3f}%\n"
                f"  depth     : {depth_values}\n"
                f"  expand    : {elasticity_ratio}\n"
                f"  width_idx : {width_multiplier}\n"
                f"{'-'*70}"
            )

        # Extract a fixed-architecture standalone subnet with the supernet's
        # pretrained weights copied in (preserve_weight=True).
        full_net.set_active_subnet(
            d=depth_values, e=elasticity_ratio, w=width_multiplier
        )
        subnet = full_net.get_active_subnet(preserve_weight=True)
        subnet.cuda()

        # Sync the extracted subnet weights across all ranks (rank-0 did the
        # extraction and weight copy; other ranks need the same weights).
        if dist.is_initialized():
            for param in subnet.parameters():
                dist.broadcast(param.data, src=0)
            for buf in subnet.buffers():
                dist.broadcast(buf, src=0)

        subnet_path = os.path.join(args.subnet_out_dir, f"subnet_{subnet_id}")
        os.makedirs(subnet_path, exist_ok=True)

        # DistributedRunManager with init=False so it does NOT reinitialise
        # the weights we just copied from the supernet.
        run_manager = DistributedRunManager(
            subnet_path,
            subnet,
            run_config,
            backward_steps=1,   # no gradient accumulation for a fixed subnet
            is_root=is_root,
            init=False,         # preserve pretrained weights!
        )
        run_manager.save_config()

        # Fine-tune.
        run_manager.train(subnet_args, warmup_epochs=0, warmup_lr=args.subnet_lr)

        # Final validation on the test split.
        _, val_loss, val_top1, val_top5 = run_manager.validate_all_resolution(
            is_test=True
        )
        final_acc = list_mean(val_top1)

        # Save a clean, standalone state-dict for direct serving use.
        # This file can be loaded with: torch.load(path)["state_dict"]
        if is_root:
            standalone_path = os.path.join(subnet_path, f"subnet_{subnet_id}_final.pth.tar")
            torch.save({"state_dict": subnet.state_dict()}, standalone_path)
            gap = final_acc - expected_acc
            print(
                f"\nSubnet {subnet_id} done.\n"
                f"  Achieved : {final_acc:.3f}%\n"
                f"  Expected : {expected_acc:.3f}%\n"
                f"  Gap      : {gap:+.3f}%\n"
                f"  Saved to : {standalone_path}"
            )

        results.append(
            {
                "id": subnet_id,
                "expected": expected_acc,
                "achieved": final_acc,
            }
        )

    # Summary table.
    if is_root:
        print(f"\n{'='*70}")
        print("All subnets trained.  Summary:")
        print(f"  {'ID':>3}  {'Expected':>10}  {'Achieved':>10}  {'Gap':>8}")
        print("  " + "-" * 38)
        for r in results:
            gap = r["achieved"] - r["expected"]
            flag = "  <-- check!" if abs(gap) > 2.0 else ""
            print(
                f"  {r['id']:>3}  {r['expected']:>10.3f}  "
                f"{r['achieved']:>10.3f}  {gap:>+8.3f}{flag}"
            )
        print(f"{'='*70}\n")

    return results


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

    # -----------------------------------------------------------------------
    # Branch: individual subnet fine-tuning
    # -----------------------------------------------------------------------
    if args.train_subnets:
        train_individual_subnets(args, run_config, is_root, num_gpus)
        if dist.is_initialized():
            dist.destroy_process_group()
        exit(0)

    # -----------------------------------------------------------------------
    # Branch: full OFA progressive-shrinking supernet training
    # -----------------------------------------------------------------------
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
