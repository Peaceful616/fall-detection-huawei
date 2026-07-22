"""学生模型 v3：YOLOv8s backbone（全解冻）+ 时空 Adapter v3 + 1D-CNN 头 + 姿态辅助分支

参数预算（v3）：
- backbone YOLOv8s ~5.08M（全解冻，unfreeze_from=0）
- adapter v3 ~1.68M（6 层 STResBlock + SE）
- head v3 ~3.49M（4 层 + 残差 + c_out=1024）
- 姿态辅助 ~0.05M（推理期不部署）
- 总参数 ~10.30M（≤20M 硬指标达标，余量 48%）
"""
import torch
import torch.nn as nn

from .backbones import build_backbone
from .adapter import (
    SpatioTemporalAdapterV3,
    CNN1DClassifier,
    PoseAuxHead,
)


class StudentNetV3(nn.Module):
    """端到端学生模型 v3

    输入: (B, T, 3, H, W) - 视频片段，T 帧
    输出:
        logits: (B, num_classes)          ← 主任务
        feat:   (B, C_out, T)             ← 供蒸馏
        backbone_feat: (B, T, C, H, W)   ← 供关系蒸馏
        aux_kp: (B, T, 17, 2)             ← 姿态辅助（仅训练期）
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # backbone（全解冻，unfreeze_from=0）
        self.backbone = build_backbone(cfg)
        # 时空 Adapter v3（6 层 STResBlock + SE）
        self.adapter = SpatioTemporalAdapterV3(
            c_in=cfg.adapter_c_in,
            c_mid=cfg.adapter_c_mid,
            c_out=cfg.adapter_c_out,
            temporal_kernel=cfg.adapter_temporal_kernel,
            depth=cfg.adapter_depth,
            residual=cfg.adapter_residual,
        )
        # 1D-CNN 分类头
        self.head = CNN1DClassifier(
            c_in=cfg.adapter_c_out,
            c_hidden=cfg.head_c_hidden,
            c_out=cfg.head_c_out,
            num_classes=cfg.num_classes,
            layers=cfg.head_layers,
            residual=cfg.head_residual,
        )
        # 姿态辅助分支（训练期使用）
        self.aux_kp_enabled = cfg.aux_kp_enabled
        if self.aux_kp_enabled:
            self.aux_kp_head = PoseAuxHead(
                c_in=cfg.adapter_c_out,
                num_kp=cfg.aux_kp_num,
            )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, 3, H, W)
        """
        B, T = x.shape[:2]
        # 1. backbone 逐帧提取特征
        feats = self.backbone(x)
        feat = feats["P4"]        # (B*T, C_in, H/16, W/16)
        _, C, H, W = feat.shape
        feat = feat.view(B, T, C, H, W)

        # 2. 时空 Adapter v3
        adapter_out = self.adapter(feat)  # (B, C_out, T)

        # 3. 1D-CNN 分类头
        logits = self.head(adapter_out)  # (B, num_classes)

        out = {
            "logits": logits,
            "feat": adapter_out,
            "backbone_feat": feat,
        }

        # 4. 姿态辅助分支（训练期）
        if self.aux_kp_enabled and self.training:
            out["aux_kp"] = self.aux_kp_head(adapter_out)

        return out

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters()
                                if p.requires_grad)
        adapter = sum(p.numel() for p in self.adapter.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        aux_kp = (sum(p.numel() for p in self.aux_kp_head.parameters())
                 if self.aux_kp_enabled else 0)
        # 推理期只部署 backbone + adapter + head（不含 aux_kp）
        infer_total = backbone + adapter + head
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "backbone_trainable": backbone_trainable,
            "adapter": adapter,
            "head": head,
            "aux_kp": aux_kp,
            "infer_total": infer_total,
            "total_M": round(total / 1e6, 2),
            "infer_M": round(infer_total / 1e6, 2),
            "fp32_MB": round(infer_total * 4 / 1e6, 2),
            "int8_MB": round(infer_total / 1e6, 2),
        }


def build_student(cfg):
    return StudentNetV3(cfg)


# 向后兼容
StudentNet = StudentNetV3
