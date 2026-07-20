"""学生模型：YOLOv8n backbone（冻结）+ 时空 Adapter + 1D-CNN 分类头

参数预算：
- backbone 3.2M (冻结，不计入可训练)
- adapter ~2M
- head ~0.3M
- 总可训练 ~2.3M，总参数 5.5M
"""
import torch
import torch.nn as nn

from .backbones import build_backbone
from .adapter import SpatioTemporalAdapter, CNN1DClassifier


class StudentNet(nn.Module):
    """端到端学生模型

    输入: (B, T, 3, H, W) - 视频片段，T 帧
    输出:
        logits: (B, num_classes)
        feat:   (B, C_out, T) - 供蒸馏用
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.adapter = SpatioTemporalAdapter(
            c_in=cfg.adapter_c_in,
            c_mid=cfg.adapter_c_mid,
            c_out=cfg.adapter_c_out,
            temporal_kernel=cfg.adapter_temporal_kernel
        )
        self.head = CNN1DClassifier(
            c_in=cfg.adapter_c_out,
            c_hidden=cfg.head_c_hidden,
            c_out=cfg.head_c_out,
            num_classes=cfg.num_classes
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, 3, H, W)
        """
        B, T = x.shape[:2]
        # 1. backbone 逐帧提取特征（冻结）
        feats = self.backbone(x)  # dict{P3/P4/P5}
        feat = feats["P4"]        # (B*T, C_in, H/16, W/16)
        _, C, H, W = feat.shape
        feat = feat.view(B, T, C, H, W)

        # 2. 时空 Adapter
        adapter_out = self.adapter(feat)  # (B, C_out, T)

        # 3. 1D-CNN 分类头
        logits = self.head(adapter_out)  # (B, num_classes)

        return {
            "logits": logits,
            "feat": adapter_out,        # 供特征蒸馏
            "backbone_feat": feat       # 供关系蒸馏
        }

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        adapter = sum(p.numel() for p in self.adapter.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "adapter": adapter,
            "head": head,
            "total_M": round(total / 1e6, 2),
            "fp32_MB": round(total * 4 / 1e6, 2),
            "int8_MB": round(total / 1e6, 2),
        }


def build_student(cfg):
    return StudentNet(cfg)
