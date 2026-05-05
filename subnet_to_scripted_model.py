import argparse
import json
import os

import torch
import torch.jit

from ofa.imagenet_classification.elastic_nn.networks import OFAResNets

LATENCY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "latency_curves_supernet_resnet_A40_with_stages_29apr26.json",
)


def load_latency_config():
    with open(LATENCY_CONFIG_PATH, "r") as f:
        data = json.load(f)
    return {m["id"]: m for m in data["models"]}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    return None


def main(checkpoints_root):
    latency_cfg = load_latency_config()
    for subdir in os.listdir(checkpoints_root):
        if not subdir.startswith("subnet_"):
            continue
        try:
            subnet_id = int(subdir.split("_")[1])
        except Exception:
            print(f"Skipping {subdir}: cannot parse subnet id")
            continue
        if subnet_id not in latency_cfg:
            print(f"No config for subnet id {subnet_id}")
            continue
        subdir_path = os.path.join(checkpoints_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        ckpt_files = [f for f in os.listdir(subdir_path) if f.endswith(".pth.tar")]
        if not ckpt_files:
            print(f"No checkpoint found in {subdir_path}")
            continue
        ckpt_path = os.path.join(subdir_path, ckpt_files[0])
        print(f"Processing {ckpt_path}")

        subnet_cfg = latency_cfg[subnet_id]["subnet_dimension"]
        depth_values = subnet_cfg["depth_values"]
        elasticity_ratio = subnet_cfg["elasticity_ratio"]
        width_multiplier = subnet_cfg["width_multiplier"]

        full_net = OFAResNets()
        full_net.set_active_subnet(
            d=depth_values,
            e=elasticity_ratio,
            w=width_multiplier,
        )
        subnet = full_net.get_active_subnet(preserve_weight=False)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        if state_dict is None:
            raise RuntimeError(f"Could not find state_dict in {ckpt_path}")
        subnet.load_state_dict(state_dict)

        subnet.eval()
        scripted_model = torch.jit.script(subnet)

        pt_path = os.path.join(subdir_path, f"{subdir}_final_model.pt")
        scripted_model.save(pt_path)
        print(f"Saved scripted model to {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OFA subnet checkpoints to TorchScript .pt files."
    )
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default="/home/dgarg39/ofa_checkpoints/finetuned_subnets",
        help="Root directory containing subnet checkpoint subdirectories.",
    )
    args = parser.parse_args()
    main(args.checkpoints_root)
