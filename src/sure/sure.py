from ..utils.misc import detect_NaN
from .head.fine_matching import FineMatching
from .head.coarse_matching import CoarseMatching
from .neck.neck import CIM
from .backbone.backbone import CovNextV2_nano,ResNet18,RepVGG_8_1_align
from einops.einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import rearrange
# from src.sure.mamba_module import JointMamba
from src.sure.utils.utils import normalize_keypoints,KeypointEncoder_wo_score
torch.set_float32_matmul_precision("highest")  # highest (defualt) high medium
from src.sure.neck.loftr_module.transformer import LocalFeatureTransformer

import time

from kornia.utils import create_meshgrid


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class SURE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.local_resolution = self.config["local_resolution"]
        self.bi_directional_refine = self.config["fine"]["bi_directional_refine"]
        self.deploy = self.config["deploy"]
        self.topk = config["coarse"]["topk"]
        self.d_model_c=256
        # Modules
        # self.backbone = CovNextV2_nano()
        # self.backbone = ResNet18(config)
        self.backbone = RepVGG_8_1_align()
        # self.neck = CIM(config)
        # self.joint_mamba = JointMamba(self.d_model_c, 4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True)
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])

        # self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        # self.FPN = InterFPN_SE()
        self.FPN = DetailEnhancedFPN()
        self.coarse_matching = CoarseMatching(config)
        self.fine_matching = FineMatching(config)

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        synchronize_cuda()
        s=time.time()
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feats_c = ret_dict['feats_c']
            data.update({
                'feats_x2': ret_dict['feats_x2'],
                'feats_x1': ret_dict['feats_x1'],
            })
            synchronize_cuda()
            # s1 = time.time()
            feat_f = self.FPN(ret_dict['feats_x1'], ret_dict['feats_x2'], feats_c)
            feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)

            (feat_c0, feat_c1) = feats_c.split(data['bs'])
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat_c0 = ret_dict0['feats_c']
            feat_c1 = ret_dict1['feats_c']

            feat_f0 = self.FPN(
                ret_dict0['feats_x1'], ret_dict0['feats_x2'], feat_c0
            )
            feat_f1 = self.FPN(
                ret_dict1['feats_x1'], ret_dict1['feats_x2'], feat_c1
            )

            data.update({
                'feats_x2_0': ret_dict0['feats_x2'],
                'feats_x1_0': ret_dict0['feats_x1'],
                'feats_x2_1': ret_dict1['feats_x2'],
                'feats_x1_1': ret_dict1['feats_x1'],
            })



        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]
        synchronize_cuda()
        s2 = time.time()

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        if (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            detect_NaN(feat_c0, feat_c1)
        synchronize_cuda()
        s3=time.time()

        # 2.  Feature Interaction & Multi-Scale Fusion

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_c0.shape[2:] * self.config["local_resolution"],
                "hw1_f": feat_c1.shape[2:] * self.config["local_resolution"],
            }
        )
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")
        feat_f0 = rearrange(feat_f0, "n c h w -> n (h w) c")
        feat_f1 = rearrange(feat_f1, "n c h w -> n (h w) c")

        # detect NaN during mixed precision training
        if self.config["mp"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        # 3. Coarse-Level Matching
        conf_matrix = self.coarse_matching(
            feat_c0,
            feat_c1,
            data,
            mask_c0=(
                mask_c0.view(mask_c0.size(0), -
                             1) if mask_c0 is not None else mask_c0
            ),
            mask_c1=(
                mask_c1.view(mask_c1.size(0), -
                             1) if mask_c1 is not None else mask_c1
            ),
        )
        synchronize_cuda()
        s4=time.time()

        # 4. Fine-Level Matching
        K0 = data["i_ids"].shape[0] // data["bs"]
        K1 = data["j_ids"].shape[0] // data["bs"]
        feat_f0 = feat_f0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_f1 = feat_f1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)
        feat_c0 = feat_c0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_c1 = feat_c1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)

        if self.bi_directional_refine:
            # Bidirectional Refinement
            self.fine_matching(
                torch.cat([feat_f0, feat_f1], dim=1),
                torch.cat([feat_f1, feat_f0], dim=1),
                torch.cat([feat_c0, feat_c1], dim=1),
                torch.cat([feat_c1, feat_c0], dim=1),
                data,
            )
        else:
            self.fine_matching(
                feat_f0, feat_f1,
                feat_c0, feat_c1, data)
        synchronize_cuda()
        # s5=time.time()
        # t=s5-s
        # print("total:",s5-s)
        # print("backbone:",(s1-s))
        # print("FPN:",(s2-s1))
        # print("loftr:",(s3-s2))
        # print("coarse:",(s4-s3))
        # print("fine:",(s5-s4))


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
#
# try:
#     from mmcv.ops import DeformConv2d
# except ImportError:
#     raise ImportError("Please install mmcv-full to use DeformConv2d.")
#
#
# class InterFPN_GatedDeform(nn.Module):
#     def __init__(self, in_channels: int=128, out_channels: int=256, kernel_size: int = 3):
#         """
#         使用低层特征引导的门控 + Deformable 卷积融合模块。
#         feat_4 提供 offset（细节引导），同时提供 gate。
#         """
#         super().__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#
#         # 1×1 降通道
#         self.reduce_l = nn.Conv2d(128, out_channels, kernel_size=1)
#         self.reduce_h = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#
#         # Gate 注意力通道（用低层生成一个注意力 mask）
#         self.gate = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // 2, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # Offset 预测器（来自低层）
#         offset_channels = 2 * kernel_size * kernel_size
#         self.offset_conv = nn.Conv2d(out_channels, offset_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#
#         # Deformable conv 融合（作用在 concat 后特征）
#         self.fuse_deform = DeformConv2d(out_channels * 2, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, feat_4: torch.Tensor, feat_8: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             feat_4: [B, C, H4, W4] 高分辨率特征（细节）
#             feat_8: [B, C, H8, W8] 低分辨率特征（语义）
#
#         Returns:
#             [B, out_channels, H8, W8] 融合后特征
#         """
#         B = feat_4.size(0)
#         feat_4 = self.reduce_l(feat_4)                      # [B, C_out, H4, W4]
#         feat_4 = F.adaptive_avg_pool2d(feat_4, feat_8.shape[-2:])  # → [B, C_out, H8, W8]
#         feat_8 = self.reduce_h(feat_8)                      # [B, C_out, H8, W8]
#
#         # Gate 控制两者权重（由低层特征生成）
#         gate_mask = self.gate(feat_4)                       # [B, 1, H, W] ∈ (0,1)
#         gated_feat = gate_mask * feat_8 + (1 - gate_mask) * feat_4
#
#         # 低层特征生成 offset
#         offset = self.offset_conv(feat_4)                   # [B, 2*K*K, H, W]
#
#         # 拼接 gated + feat_4（也可用 gated + feat_8）
#         fused = torch.cat([feat_4, gated_feat], dim=1)      # [B, 2C_out, H, W]
#         out = self.fuse_deform(fused, offset)               # [B, C_out, H, W]
#         return self.relu(out)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterFPN_SE(nn.Module):
    def __init__(self, in_channels_l=128, in_channels_h=256, out_channels=256, reduction=16):
        """
        feat_4 通道128，feat_8 通道256，输出256通道
        SE注意力融合模块
        """
        super().__init__()
        self.reduce_l = nn.Conv2d(in_channels_l, out_channels, kernel_size=1)
        self.reduce_h = nn.Conv2d(in_channels_h, out_channels, kernel_size=1)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B x C x 1 x 1
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_4, feat_8):
        feat_4 = self.reduce_l(feat_4)  # B x 256 x H4 x W4
        feat_4 = F.adaptive_avg_pool2d(feat_4, feat_8.shape[-2:])  # 下采样到 feat_8 大小
        feat_8 = self.reduce_h(feat_8)  # B x 256 x H8 x W8

        fused = feat_4 + feat_8  # 通道数一致，可以直接相加

        se_weight = self.se(fused)  # B x 256 x 1 x 1

        fused = fused * se_weight

        out = self.fuse_conv(fused)
        return out

class DetailEnhancedFPN(nn.Module):
    def __init__(self, in_dims=[64, 128, 256], out_dim=256):
        super().__init__()
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, 1) for in_dim in in_dims
        ])

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 保留细节路径（来自 f2）
        self.detail_path = nn.Sequential(
            nn.Conv2d(in_dims[0], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f2, f4, f8):
        f2_down = F.adaptive_avg_pool2d(self.reduce_convs[0](f2), f8.shape[-2:])
        f4_down = F.adaptive_avg_pool2d(self.reduce_convs[1](f4), f8.shape[-2:])
        f8_proj = self.reduce_convs[2](f8)

        fused = torch.cat([f2_down, f4_down, f8_proj], dim=1)
        fused = self.fuse_conv(fused)

        # 加入细节增强（从 f2 提取的高分辨率信息）
        detail_feat = F.adaptive_avg_pool2d(self.detail_path(f2), f8.shape[-2:])
        out = fused + detail_feat
        return out
#
# from loguru import logger
#
# def detect_NaN(feat_0, feat_1):
#     logger.info(f'NaN detected in feature')
#     logger.info(f"#NaN in feat_0: {torch.isnan(feat_0).int().sum()}, #NaN in feat_1: {torch.isnan(feat_1).int().sum()}")
#     feat_0[torch.isnan(feat_0)] = 0
#     feat_1[torch.isnan(feat_1)] = 0
