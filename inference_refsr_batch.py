import sys

sys.path.append("/mnt/massive/wangce/.cache/torch/hub/facebookresearch_dinov2_main")
import shutil
import os
from argparse import ArgumentParser, Namespace
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ldm.xformers_state import disable_xformers
from img_utils.common import instantiate_from_config, load_state_dict
from torch.utils.data import DataLoader
from torch import nn
import numpy as np


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # TODO: add help info for these options
    parser.add_argument(
        "--ckpt",
        default="step=372999-val_psnr=20.864.ckpt",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--config",
        default="configs/model/refsr_dino.yaml",
        type=str,
        help="model config path",
    )
    parser.add_argument(
        "--val_config", type=str, default="configs/dataset/reference_sr_test.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/exp",
    )
    parser.add_argument("--global_ref_scale", type=float, default=1.0, help="global scalar scaling factor for reference")
    parser.add_argument(
        "--local_ref",
        type=bool,
        default=False,
        help="Whether to use local reference scaling"
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cpu", "cuda", "mps"]
    )

    return parser.parse_args()


def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


def split_result(input_folder):
    # 定义输出文件夹路径
    output_folder_hq = input_folder + "/hq"
    output_folder_samples = input_folder + "/samples"
    output_folder_lq = input_folder + "/lq"
    output_folder_ref = input_folder + "/ref"

    # 确保输出文件夹存在，如果不存在就创建
    os.makedirs(output_folder_hq, exist_ok=True)
    os.makedirs(output_folder_samples, exist_ok=True)
    os.makedirs(output_folder_lq, exist_ok=True)
    os.makedirs(output_folder_ref, exist_ok=True)

    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为PNG格式
        if filename.lower().endswith(".png") and os.path.isfile(file_path):
            # 检查文件名是否以hq结尾
            if "hq" in filename:
                filename = filename.replace("_hq", "")
                # 如果是，将文件复制到hq文件夹，并在目标路径中包含文件名
                shutil.move(file_path, os.path.join(output_folder_hq, filename))
            # 检查文件名是否以samples结尾
            elif "samples" in filename:
                filename = filename.replace("_samples", "")
                # 如果是，将文件复制到samples文件夹，并在目标路径中包含文件名
                shutil.move(file_path, os.path.join(output_folder_samples, filename))
            # 检查文件名是否以samples结尾
            elif "lq" in filename:
                filename = filename.replace("_lq", "")
                # 如果是，将文件复制到lq文件夹，并在目标路径中包含文件名
                shutil.move(file_path, os.path.join(output_folder_lq, filename))
            # 检查文件名是否以samples结尾
            elif "ref" in filename:
                filename = filename.replace("_ref", "")
                # 如果是，将文件复制到ref文件夹，并在目标路径中包含文件名
                shutil.move(file_path, os.path.join(output_folder_ref, filename))

    print("PNG文件已复制完成。")


class LogImagesWrapper(nn.Module):
    def __init__(self, model):
        super(LogImagesWrapper, self).__init__()
        self.model = model

    def forward(self, val_data):
        return self.model.log_images(val_data)
        # return self.model.encode(val_data).sample()


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    val_dataset = instantiate_from_config(OmegaConf.load(args.val_config)["dataset"])
    val_dataloader = DataLoader(
        dataset=val_dataset, **(OmegaConf.load(args.val_config)["data_loader"])
    )

    model = instantiate_from_config(OmegaConf.load(args.config))

    static_dic = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, static_dic, strict=True)

    model.freeze()
    model.to(args.device)
    model.eval()

    val_dataset = instantiate_from_config(OmegaConf.load(args.val_config)["dataset"])
    val_dataloader = DataLoader(
        dataset=val_dataset, **(OmegaConf.load(args.val_config)["data_loader"])
    )

    if args.global_ref_scale != 1:
        sim_lamuda = args.global_ref_scale
    elif args.local_ref:
        sim_lamuda = torch.ones((480, 480)).to(args.device)
    else:
        sim_lamuda = None
    with torch.cuda.amp.autocast():
        for idx, val_data in enumerate(val_dataloader):
            model.validation_inference(
                val_data, idx, args.output, sim_lamuda=sim_lamuda
            )
    split_result(args.output)


if __name__ == "__main__":
    main()
    # split_result("./results/test-anchor-1.3-25step")
