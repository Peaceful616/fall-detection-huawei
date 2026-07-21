"""bafd6 检测模型：基于学生 backbone + Adapter + 检测头

复用 YOLOv8s backbone + Adapter v3，新增检测头：
- 输入: (B, 1, 3, H, W) 伪视频化单帧
- 输出:
    - bbox: (B, num_anchors, 4) [cx, cy, w, h] 归一化
    - cls:  (B, num_anchors, num_classes) softmax
    - feat: (B, C_out, 1) 供蒸馏

简化设计：
- num_anchors = 1（每张图预测 1 个 bbox，因为是单目标跌倒）
- 实际多目标场景可改为 NMS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .adapter import SpatioTemporalAdapterV2


class DetectionHead(nn.Module):
    """检测头：输出 bbox + 类别

    输入: (B, C, T=1) - Adapter 输出
    输出:
        bbox: (B, num_anchors, 4) [cx, cy, w, h]
        cls:  (B, num_anchors, num_classes)
    """

    def __init__(self, c_in: int = 128, num_anchors: int = 3,
                 num_classes: int = 1):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # 共享特征 + anchor-specific 输出
        self.shared = nn.Sequential(
            nn.Conv1d(c_in, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        # bbox 回归（4 维）
        self.bbox_head = nn.Conv1d(256, num_anchors * 4, 1)
        # 类别预测（num_classes 维，含 background）
        self.cls_head = nn.Conv1d(256, num_anchors * num_classes, 1)
        # objectness（anchor 是否有目标）
        self.obj_head = nn.Conv1d(256, num_anchors, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        h = self.shared(x)  # (B, 256, T)
        B, _, T = h.shape
        bbox = self.bbox_head(h).view(B, self.num_anchors, 4, T)
        cls = self.cls_head(h).view(B, self.num_anchors, self.num_classes, T)
        obj = self.obj_head(h).view(B, self.num_anchors, T)
        # T=1 时取首帧
        bbox = bbox[..., 0]   # (B, num_anchors, 4)
        cls = cls[..., 0]     # (B, num_anchors, num_classes)
        obj = obj[..., 0]     # (B, num_anchors)
        return {"bbox": bbox, "cls": cls, "obj": obj}


class FallDetector(nn.Module):
    """端到端跌倒检测器

    输入: (B, T=1, 3, H, W) 伪视频化单帧
    输出: bbox + cls + obj + feat（供蒸馏）
    """

    def __init__(self, cfg, num_anchors: int = 3, num_classes: int = 1):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.adapter = SpatioTemporalAdapterV2(
            c_in=cfg.adapter_c_in,
            c_mid=cfg.adapter_c_mid,
            c_out=cfg.adapter_c_out,
            temporal_kernel=cfg.adapter_temporal_kernel,
            depth=cfg.adapter_depth,
            residual=cfg.adapter_residual,
        )
        self.head = DetectionHead(
            c_in=cfg.adapter_c_out,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T=1, 3, H, W)
        B, T = x.shape[:2]
        feats = self.backbone(x)  # dict
        feat = feats["P4"]
        _, C, H, W = feat.shape
        feat = feat.view(B, T, C, H, W)
        adapter_out = self.adapter(feat)  # (B, C_out, T=1)
        det = self.head(adapter_out)
        return {
            **det,
            "feat": adapter_out,
            "backbone_feat": feat,
        }

    @torch.no_grad()
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": round(total / 1e6, 2),
            "fp32_MB": round(total * 4 / 1e6, 2),
            "int8_MB": round(total / 1e6, 2),
        }


def build_detector(cfg):
    return FallDetector(cfg, num_anchors=3, num_classes=1)
