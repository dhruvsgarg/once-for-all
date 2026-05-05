import torch
import torch.jit
import os
import argparse
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets

def main(checkpoints_root):
	# Iterate over all subdirectories in the checkpoints_root
	for subdir in os.listdir(checkpoints_root):
		subdir_path = os.path.join(checkpoints_root, subdir)
		if not os.path.isdir(subdir_path):
			continue
		# Find the checkpoint file (assume it ends with .pth.tar)
		ckpt_files = [f for f in os.listdir(subdir_path) if f.endswith('.pth.tar')]
		if not ckpt_files:
			print(f"No checkpoint found in {subdir_path}")
			continue
		ckpt_path = os.path.join(subdir_path, ckpt_files[0])
		print(f"Processing {ckpt_path}")

		# 1. Load model
		model = OFAResNets()
		checkpoint = torch.load(ckpt_path, map_location='cpu')
		# model.load_state_dict(checkpoint['model_state']) # or just checkpoint
		try:
			model.load_state_dict(checkpoint)
		except Exception:
			model.load_state_dict(checkpoint['model_state'])

		# 2. Set to eval mode
		model.eval()

		# 3. Script the model
		scripted_model = torch.jit.script(model)

		# 4. Save the scripted model
		pt_path = os.path.join(subdir_path, f"{subdir}_final_model.pt")
		scripted_model.save(pt_path)
		print(f"Saved scripted model to {pt_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert OFA subnet checkpoints to TorchScript .pt files.")
	parser.add_argument(
		"--checkpoints_root",
		type=str,
		default="/home/dgarg39/ofa_checkpoints/finetuned_subnets",
		help="Root directory containing subnet checkpoint subdirectories."
	)
	args = parser.parse_args()
	main(args.checkpoints_root)
