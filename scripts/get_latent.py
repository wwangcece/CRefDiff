import sys

import einops

sys.path.append(".")
from argparse import ArgumentParser
from typing import Dict
import os
import torch
from omegaconf import OmegaConf
from img_utils.common import instantiate_from_config, load_state_dict
from img_utils.common import instantiate_from_config
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--cldm_config", type=str, default="configs/model/refsr.yaml")
parser.add_argument(
    "--weight_path",
    type=str,
    default="experiments/anchor-ref/lightning_logs/version_0/checkpoints/step=158999-val_lpips=0.321.ckpt",
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--input",
    type=str,
    default="/mnt/massive/wangce/RefSR_x10/dataset/val/HR",
)
parser.add_argument(
    "--output",
    type=str,
    default="/mnt/massive/wangce/RefSR_x10/dataset/val/HR_latent",
)
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

model = instantiate_from_config(OmegaConf.load(args.cldm_config))
load_state_dict(model, torch.load(args.weight_path, map_location="cpu"), strict=True)
model.freeze()
model.to(args.device)

with tqdm(total=len(os.listdir(args.input)), desc="Processing") as pbar:
    for img_name in os.listdir(args.input):
        img_path = os.path.join(args.input, img_name)
        out_path = os.path.join(args.output, img_name.split(".")[0] + ".npy")
        image = np.array(Image.open(img_path).convert("RGB"))
        input = torch.tensor(
            np.stack([image]) / 255.0, dtype=torch.float32, device=model.device
        ).clamp_(0, 1)
        input = einops.rearrange(input, "n h w c -> n c h w").contiguous()
        input = input * 2 - 1
        encoder_posterior = model.encode_first_stage(input)
        z = (
            model.get_first_stage_encoding(encoder_posterior)
            .detach()
            .squeeze()
            .cpu()
            .numpy()
        )
        np.save(out_path, z)
        pbar.update()
