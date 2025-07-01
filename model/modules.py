import cmath
import math
from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import random

"""
    CNN-SR
"""


def cosine_attention_map(tensor1, tensor2, eps=1e-8):
    """
    输入:
        tensor1, tensor2: 形状为 (B, C, H, W) 的两个张量
    输出:
        cosine_distance_map: 形状为 (B, 1, H, W) 的注意力图
    """
    B, C, H, W = tensor1.shape

    # 展平通道维，准备计算每个像素位置的余弦相似度
    x1 = tensor1.view(B, C, -1)  # (B, C, H*W)
    x2 = tensor2.view(B, C, -1)

    # L2 归一化
    x1_norm = F.normalize(x1, p=2, dim=1)  # (B, C, H*W)
    x2_norm = F.normalize(x2, p=2, dim=1)

    # 对应位置点积（余弦相似度）
    cos_sim = torch.sum(x1_norm * x2_norm, dim=1, keepdim=True)  # (B, 1, H*W)

    # reshape 回 (B, 1, H, W)
    return cos_sim.view(B, 1, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out


class Resblock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, padding=1):
        super(Resblock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        identity = x
        out = self.res_block(x)
        return out + identity


class Downsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                in_channels=in_channels * scale * scale,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.downsample(x)


class SR_Encoder(pl.LightningModule):
    def __init__(self, out_channel=8, in_channel=3):
        super(SR_Encoder, self).__init__()

        self.first_layer_sr = nn.Conv2d(
            in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  ##128 128 64
            nn.LeakyReLU(),
        )

        self.layer2_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # 64 64 128
            nn.LeakyReLU(),
        )

        self.layer3_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
            ),  # 32 32 256
            nn.LeakyReLU(),
        )

        # (3)out
        self.last_linear = nn.Conv2d(256, out_channel, 3, bias=False, padding=1)

    def forward(self, sr):
        # (1)cnn encoder
        # b 3 256 256 -> b 256 32 32
        sr_cond = self.first_layer_sr(sr)
        sr_cond = self.layer1_sr(sr_cond)
        sr_cond = self.layer2_sr(sr_cond)
        sr_cond = self.layer3_sr(sr_cond)
        out = self.last_linear(sr_cond)
        # 4 32 32
        return out


class LocalCrossAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ln_ref = nn.LayerNorm(dim)
        self.ln_sr = nn.LayerNorm(dim)

    def forward(self, sr, ref, return_cos_sim_map=False, sim_lamuda=1):
        B, C, H, W = sr.shape
        assert (
            H % self.window_size == 0 and W % self.window_size == 0
        ), "H and W must be divisible by window_size"

        attn = (cosine_attention_map(sr, ref) + 1) / 2  # B 1 H W
        # Step 1: unfold to non-overlapping windows
        sr_windows = F.unfold(
            sr, kernel_size=self.window_size, stride=self.window_size
        )  # B, C*win*win, N_win
        ref_windows = F.unfold(
            ref, kernel_size=self.window_size, stride=self.window_size
        )

        # Now shape: B, C*win*win, Num_windows → reshape to B*N, win*win, C
        B, _, N = sr_windows.shape
        win_area = self.window_size * self.window_size

        sr_windows = (
            sr_windows.transpose(1, 2).reshape(B * N, C, win_area).permute(0, 2, 1)
        )  # [B*N, win_area, C]
        ref_windows = (
            ref_windows.transpose(1, 2).reshape(B * N, C, win_area).permute(0, 2, 1)
        )

        # Step 2: cross-attention: Q=sr, K/V=ref
        sr_windows = self.ln_sr(sr_windows)
        ref_windows = self.ln_ref(ref_windows)
        fused_windows, _ = self.attn(
            query=sr_windows, key=ref_windows, value=ref_windows
        )  # [B*N, win_area, C]

        # Step 3: reshape back
        fused_windows = (
            fused_windows.permute(0, 2, 1).reshape(B, N, C * win_area).transpose(1, 2)
        )  # B, C*win_area, N
        out = F.fold(
            fused_windows,
            output_size=(H, W),
            kernel_size=self.window_size,
            stride=self.window_size,
        )  # B, C, H, W

        if self.training and random.random() < 0.2:
            sim_lamuda = 0
        
        if isinstance(sim_lamuda, float):
            attn = (attn * sim_lamuda).clip(0, 1)
        elif isinstance(sim_lamuda, torch.Tensor):
            sim_lamuda = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(attn.shape[2], attn.shape[3]),
            )
            attn = (attn * sim_lamuda).clip(0, 1)
        
        out = attn * out + (1 - attn) * sr

        if not return_cos_sim_map:
            return out
        else:
            return out, attn


class MaskAttention(nn.Module):
    def __init__(self, channels):
        super(MaskAttention, self).__init__()

        # 分别处理 sr 和 ref 的卷积模块（共享结构）
        self.sr_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ref_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 拼接后生成注意力图
        self.attention = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sr, ref, sim_lamuda=1, return_learned_sim_map=False):
        sr_feat = self.sr_conv(sr)  # B C H W
        ref_feat = self.ref_conv(ref)  # B C H W

        fused = torch.cat([sr_feat, ref_feat], dim=1)  # B 2C H W
        attn = self.attention(fused)  # B C H W, in [0, 1]

        # 加权融合
        if self.training and random.random() < 0.2:
            sim_lamuda = 0
            
        if isinstance(sim_lamuda, float):
            attn = (attn * sim_lamuda).clip(0, 1)
        elif isinstance(sim_lamuda, torch.Tensor):
            sim_lamuda = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(attn.shape[2], attn.shape[3]),
            )
            attn = (attn * sim_lamuda).clip(0, 1)

        # attn = (attn ** sim_lamuda).clip(0, 1)
        output = attn * ref_feat + (1 - attn) * sr_feat  # B C H W
        if not return_learned_sim_map:
            return output
        else:
            return output, torch.mean(attn, dim=1, keepdim=True)

class SR_Ref_Encoder_LCA(pl.LightningModule):
    def __init__(self, out_channel=8, in_sr_channel=3, in_ref_channel=3):
        super(SR_Ref_Encoder_LCA, self).__init__()

        self.first_layer_sr = ConvBlock(
            in_channels=in_sr_channel,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 256 256 64

        self.first_layer_ref = ConvBlock(
            in_channels=in_ref_channel,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 256 256 64

        # (1) cnn encoder
        self.layer1_sr = nn.Sequential(
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 256 256 128
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 128
            Downsample(in_channels=64, scale=2),  # 128 128 128
        )

        self.layer2_sr = nn.Sequential(
            Resblock(n_feat=64, kernel_size=3, stride=1, padding=1),  # 128 128 128
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_sr = nn.Sequential(
            Resblock(n_feat=128, kernel_size=3, stride=1, padding=1),  # 64 64 256
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.layer1_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.lca1 = LocalCrossAttention(64, 8, 4)
        self.lca2 = LocalCrossAttention(128, 4, 4)
        self.lca3 = LocalCrossAttention(256, 2, 4)

        self.mask_attn1 = MaskAttention(64)
        self.mask_attn2 = MaskAttention(128)
        self.mask_attn3 = MaskAttention(256)

        self.last_linear = nn.Conv2d(512, out_channel, 1, bias=False)

    def forward(
        self,
        sr,
        ref,
        return_cos_sim_map=False,
        return_learned_sim_map=False,
        sim_lamuda=1,
    ):
        sr_cond = self.first_layer_sr(sr)
        ref_cond = self.first_layer_ref(ref)

        ref_cond = self.layer1_ref(ref_cond)
        sr_cond = self.layer1_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map1 = self.lca1(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca1(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )

        if return_learned_sim_map:
            sr_cond2, learned_map1 = self.mask_attn1(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn1(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = sr_cond1 + sr_cond2

        ref_cond = self.layer2_ref(ref_cond)
        sr_cond = self.layer2_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map2 = self.lca2(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca2(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )

        if return_learned_sim_map:
            sr_cond2, learned_map2 = self.mask_attn2(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn2(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = sr_cond1 + sr_cond2

        ref_cond = self.layer3_ref(ref_cond)
        sr_cond = self.layer3_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map3 = self.lca3(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca3(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        if return_learned_sim_map:
            sr_cond2, learned_map3 = self.mask_attn3(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn3(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = torch.cat([sr_cond1, sr_cond2], dim=1)

        out = self.last_linear(sr_cond)

        if not return_cos_sim_map and not return_learned_sim_map:
            return out
        elif return_cos_sim_map:
            return out, [cos_map1, cos_map2, cos_map3]
        elif return_learned_sim_map:
            return out, [learned_map1, learned_map2, learned_map3]

class ImplicitPromptModule(nn.Module):

    def __init__(
        self,
        image_feat_dim=1280,
        proj_dim=1024,
        num_queries=256,
        embed_dim=1024,
        num_heads=8,
    ):
        super().__init__()

        # Projector (MLP)
        self.projector = nn.Sequential(
            nn.Linear(image_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, embed_dim),
        )

        # Learnable Queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Transformer-style layers
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, image_feat, sim_lamuda=None):
        # 2. 通过 MLP projector 映射到目标维度
        vis_feat = self.projector(image_feat)  # [B, N, proj_dim]

        # 3. Learnable Queries
        B = image_feat.size(0)
        queries = self.queries.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, num_queries, embed_dim]

        # 5. Cross-Attention: queries attend to visual features
        q_norm = self.layernorm(queries)
        vis_norm = self.layernorm(vis_feat)

        if isinstance(sim_lamuda, float):
            factor = sim_lamuda
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm)
            queries = queries + factor * q_ca  # residual
        elif isinstance(sim_lamuda, torch.Tensor):
            mask = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(
                    int(math.sqrt(vis_feat.shape[1])),
                    int(math.sqrt(vis_feat.shape[1])),
                ),
            )
            mask = mask.reshape(1, -1).repeat(
                queries.shape[1], 1
            )  # [num_queries, N]
            attn_mask = mask.masked_fill(mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm, attn_mask=attn_mask)
            queries = queries + q_ca  # residual
        else:
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm)
            queries = queries + q_ca  # residual

        # 6. Feed-Forward Network
        q_norm = self.layernorm(queries)
        out = queries + self.ffn(q_norm)  # residual

        # 输出的隐式文本特征：F_txt_imp
        return out  # shape: [B, num_queries, embed_dim]
