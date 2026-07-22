"""三教师模型：SlowFast / VideoSwin / MViT

教师只在训练期使用，推理时仅部署学生网络。
三个教师均使用官方预训练权重作为 backbone，仅在最后替换分类头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ SlowFast 教师（3D-CNN，官方预训练权重）============

class SlowFastTeacher(nn.Module):
    """SlowFast 教师模型

    使用 pytorchvideo 官方 SlowFast R50 预训练权重作为 backbone，
    替换最后的分类头为当前任务的分类头。

    输入: (B, T, 3, H, W) - 视频片段
    输出:
        logits: (B, num_classes)
        feat_list: List[Tensor] - 供特征蒸馏
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            "slowfast_r50",
            pretrained=True,
        )
        # 替换最后的分类头
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, T, 3, H, W) → (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, 16, H, W)
        # 预解码只保存了 16 帧，但 SlowFast R50 8x8 需要 fast=32 / slow=8 帧。
        # 使用三线性插值把时序维度从 16 扩展到 32，避免重新预解码。
        x = F.interpolate(x, size=(32, x.size(-2), x.size(-1)),
                          mode='trilinear', align_corners=False)
        # Slow pathway: 每隔 4 帧取一帧 -> T=8
        slow = x[:, :, ::4, :, :]
        # Fast pathway: 原始帧率 -> T=32
        fast = x
        # 官方 SlowFast 输入格式是 list [slow, fast]
        logits = self.model([slow, fast])
        # 提取 logits 作为特征蒸馏用（后续可扩展为多层特征）
        return {"logits": logits, "feat_list": [logits]}


# ============ VideoSwin 教师（Video Transformer，官方预训练权重）============

class VideoSwinTeacher(nn.Module):
    """VideoSwin 教师模型

    使用 torchvision 官方 Video Swin Transformer 3D 预训练权重作为 backbone，
    替换最后的分类头为当前任务的分类头。

    输入: (B, T, 3, H, W) - 视频片段
    输出:
        logits: (B, num_classes)
        feat_list: List[Tensor] - 供特征蒸馏
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        self.model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, T, 3, H, W) → (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(x)
        return {"logits": logits, "feat_list": [logits]}


# ============ MViT 跨模态教师（官方预训练权重）============

class MViTTeacher(nn.Module):
    """MViT 教师模型

    使用 torchvision 官方 MViT v2 预训练权重作为 backbone，
    替换最后的分类头为当前任务的分类头。

    输入: (B, T, 3, H, W) - 视频片段
    输出:
        logits: (B, num_classes)
        feat_list: List[Tensor] - 供特征蒸馏
        modal_logits: (B, 2) - 模态分类（仅训练用，保持接口兼容）
    """

    def __init__(self, num_classes: int = 5, num_modal: int = 2):
        super().__init__()
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        self.model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        in_features = self.model.head[1].in_features
        self.model.head[1] = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.num_modal = num_modal

    def forward(self, x, alpha=1.0):
        # x: (B, T, 3, H, W) → (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(x)
        # 保持和旧接口兼容：返回 modal_logits（不再使用模态对抗，但兼容旧代码）
        modal_logits = torch.zeros((x.size(0), self.num_modal), device=x.device)
        return {"logits": logits, "feat_list": [logits], "modal_logits": modal_logits}


# ============ 工厂函数 ============

def build_teacher(name: str, num_classes: int = 5):
    if name == "slowfast":
        return SlowFastTeacher(num_classes)
    if name == "video_swin":
        return VideoSwinTeacher(num_classes=num_classes)
    if name == "mvit":
        return MViTTeacher(num_classes=num_classes)
    raise ValueError(f"Unknown teacher: {name}")
