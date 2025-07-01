import os
import random
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from scipy import linalg
from tqdm import tqdm


# ========== 图像加载 ==========
class ImageFolder299(Dataset):
    def __init__(self, folder, transform, sample_size):
        self.image_paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {folder}")
        self.sample_size = sample_size
        self.transform = transform

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        path = random.choice(self.image_paths)
        img = Image.open(path).convert("RGB")
        return self.transform(img)


# ========== 自定义变换 ==========
class RandomResizeOrCrop:
    def __init__(self, size=299, resize_prob=0.5):
        self.size = size
        self.resize_prob = resize_prob

    def __call__(self, img):
        if random.random() < self.resize_prob:
            return transforms.Resize((self.size, self.size))(img)
        else:
            return transforms.RandomCrop(self.size)(img)


# ========== 特征提取器 ==========
def get_inception_model(device):
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def compute_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            if batch.shape[-1] != 299:
                batch = F.interpolate(
                    batch, size=(299, 299), mode="bilinear", align_corners=False
                )
            feat = model(batch).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)


# ========== FID ==========
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real_dir",
        type=str,
        default="/mnt/massive/wangce/RefSR_x10/dataset/All_2/test_Lx/L4/HR",
        help="Path to real images",
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="/mnt/massive/wangce/RefSR_x10/DATSR-main/experiments/test/results/test_RefSR_x10_gan_v2/visualization/L4",
        help="Path to fake/generated images",
    )
    parser.add_argument(
        "--name", type=str, default='L4', help="Name of method (for printing)"
    )
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--resize_prob", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            RandomResizeOrCrop(299, args.resize_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    model = get_inception_model(device)

    dataset_real = ImageFolder299(args.real_dir, transform, args.sample_size)
    dataset_fake = ImageFolder299(args.fake_dir, transform, args.sample_size)

    loader_real = DataLoader(
        dataset_real, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    loader_fake = DataLoader(
        dataset_fake, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"➡️ [{args.name}] Extracting real features...")
    features_real = compute_features(loader_real, model, device)
    print(f"➡️ [{args.name}] Extracting fake features...")
    features_fake = compute_features(loader_fake, model, device)

    mu1, sigma1 = compute_statistics(features_real)
    mu2, sigma2 = compute_statistics(features_fake)

    fid = calculate_fid(mu1, sigma1, mu2, sigma2)
    print(f"\n✅ FID [{args.name}]: {fid:.4f}")


if __name__ == "__main__":
    main()
