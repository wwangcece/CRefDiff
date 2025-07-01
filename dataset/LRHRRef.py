from PIL import Image
import torch
from torch.utils.data import Dataset
import dataset.exps.util as Util
import numpy as np
import random
import torchvision.transforms as T

class LRHRRefDataset(Dataset):
    def __init__(
        self,
        dataroot_hr,
        dataroot_lr,
        dataroot_ref,
        split="train",
        data_len=-1,
        patch_size=None,  # 添加 patch_size 参数
        use_ColorJitter=False,
        use_gray=False,
        gt_as_ref=False
    ):
        self.data_len = data_len
        self.split = split
        self.patch_size = patch_size
        self.use_ColorJitter = use_ColorJitter
        self.gt_as_ref = gt_as_ref
        self.use_gray = use_gray

        self.lr_path = Util.get_paths_from_images(dataroot_lr)
        self.hr_path = Util.get_paths_from_images(dataroot_hr)
        self.ref_path = Util.get_paths_from_images(dataroot_ref)

        self.jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ) if use_ColorJitter else None

        dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = dataset_len
        else:
            self.data_len = min(self.data_len, dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = (
            np.asarray(Image.open(self.hr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_Ref = (
            np.asarray(Image.open(self.ref_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_LR = (
            np.asarray(Image.open(self.lr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)

        if self.gt_as_ref and random.random() < 0.3:
            img_Ref = img_HR

        # 数据增强
        [img_LR, img_Ref, img_HR] = Util.transform_augment(
            [img_LR, img_Ref, img_HR], split=self.split
        )

        # 如果启用了 ColorJitter，则对 img_Ref 进行增强
        if self.use_gray and random.random() < 0.3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img_Ref.dtype, device=img_Ref.device)
            gray = (img_Ref * weights[:, None, None]).sum(dim=0)  # shape: (H, W)
            img_Ref = gray.unsqueeze(0).repeat(3, 1, 1)
        elif self.jitter is not None:
            img_Ref_np = np.asarray(img_Ref.permute(1, 2, 0))
            img_Ref_PIL = Image.fromarray((img_Ref_np * 255).astype(np.uint8))
            img_Ref_PIL = self.jitter(img_Ref_PIL)
            img_Ref = torch.tensor(np.asarray(img_Ref_PIL).astype(np.float32) / 255.0)
            img_Ref = img_Ref.permute(2, 0, 1)

        # 随机裁剪
        if self.patch_size is not None:
            _, h, w = img_HR.shape
            ps = self.patch_size

            if h < ps or w < ps:
                raise ValueError(f"Patch size {ps} is larger than image size {h}x{w}.")

            rnd_h = random.randint(0, h - ps)
            rnd_w = random.randint(0, w - ps)

            img_HR = img_HR[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]
            img_LR = img_LR[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]
            img_Ref = img_Ref[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]

        return {
            "HR": 2 * img_HR - 1,
            "LR": 2 * img_LR - 1,
            "Ref": 2 * img_Ref - 1,
            "path": self.hr_path[index],
            "txt": "",
        }
