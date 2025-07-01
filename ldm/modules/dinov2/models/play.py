import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import cv2
import os
from tqdm import tqdm


def extract_dinov2_feature_map_from_folder(
    model_path,
    model_name,
    input_dir,
    output_dir,
    image_size=840,
    ext_list={".png", ".jpg", ".jpeg", ".bmp"},
):
    # 加载模型
    model = torch.hub.load(model_path, model_name, source="local").cuda().eval()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹中的所有图片
    image_files = [
        f for f in os.listdir(input_dir) if os.path.splitext(f)[-1].lower() in ext_list
    ]
    print(f"Found {len(image_files)} images in {input_dir}.")

    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

        # 读取并预处理图像
        image = Image.open(input_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = transform(image).unsqueeze(0).cuda()  # [1, 3, H, W]

        # 提取 patch token 特征（不含 [CLS]）
        with torch.no_grad():
            feats = model.forward_features(img_tensor)[
                "x_norm_patchtokens"
            ]  # [1, N, C]
            feats = feats.squeeze(0).cpu().numpy()  # [N, C]

        # PCA 降维到 3 通道
        pca = PCA(n_components=3)
        feats_3d = pca.fit_transform(feats)  # [N, 3]
        feats_3d -= feats_3d.min(0)
        feats_3d /= feats_3d.max(0)
        feats_3d *= 255.0
        feats_3d = feats_3d.astype(np.uint8)

        # 恢复为 2D 彩色图
        num_patches = feats_3d.shape[0]
        patch_h = patch_w = int(num_patches**0.5)
        feat_img = feats_3d.reshape(patch_h, patch_w, 3)

        # 上采样到 image_size × image_size
        feat_img = cv2.resize(
            feat_img, (image_size, image_size), interpolation=cv2.INTER_NEAREST
        )

        # 保存结果
        cv2.imwrite(output_path, cv2.cvtColor(feat_img, cv2.COLOR_RGB2BGR))


# 示例用法
if __name__ == "__main__":
    extract_dinov2_feature_map_from_folder(
        model_path="/mnt/massive/wangce/.cache/torch/hub/facebookresearch_dinov2_main",
        model_name="dinov2_vitb14",
        input_dir="/mnt/massive/wangce/RefSR_x10/dataset/val/HR_Ref",  # 替换为你的输入图像目录
        output_dir="/mnt/massive/wangce/RefSR_x10/dataset/val/HR_Ref_Dinov2",  # 输出图像保存目录
        image_size=560,
    )
