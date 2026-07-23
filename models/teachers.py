"""三教师模型：SlowFast / VideoSwin / MViT

教师只在训练期使用，推理时仅部署学生网络。
三个教师均使用官方预训练权重作为 backbone，仅在最后替换分类头。
同时暴露中间层特征供特征蒸馏与关系蒸馏使用。
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
        feat_list: List[Tensor] - 供特征蒸馏，包含全局平均池化后的特征
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            "slowfast_r50",
            pretrained=True,
        )
        # 替换最后的分类头，但保留原 projection 的输入维度用于提取特征
        self.proj_in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(self.proj_in_features, num_classes)
        # 拆出最后一层 (pool + dropout + proj) 用于后续 logits 计算
        self.final_block = self.model.blocks[-1]
        self.body_blocks = self.model.blocks[:-1]
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

        # 前向传播到 body blocks（去掉最后的 pool/proj）
        feat = self.body_blocks([slow, fast])  # list[slow_feat, fast_feat]
        # 对 slow 和 fast 做全局平均池化并拼接，作为整体特征
        pooled = []
        for f in feat:
            pooled.append(F.adaptive_avg_pool3d(f, 1).flatten(1))
        feat_vec = torch.cat(pooled, dim=1)  # (B, proj_in_features)

        logits = self.final_block(feat)  # 经过 pool + dropout + proj
        return {"logits": logits, "feat_list": [feat_vec]}


# ============ VideoSwin 教师（Video Transformer，官方预训练权重）============

class VideoSwinTeacher(nn.Module):
    """VideoSwin 教师模型

    使用 torchvision 官方 Video Swin Transformer 3D 预训练权重作为 backbone，
    替换最后的分类头为当前任务的分类头。

    输入: (B, T, 3, H, W) - 视频片段
    输出:
        logits: (B, num_classes)
        feat_list: List[Tensor] - 供特征蒸馏，包含 flatten 后的特征向量
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        self.model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        self.head_in_features = self.model.head.in_features
        self.model.head = nn.Linear(self.head_in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, T, 3, H, W) → (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # torchvision swin3d_t 结构：features + head
        # features 输出 (B, C, T', H', W')，然后 head 内部做 permute/avgpool/flatten/linear
        feat = self.model.features(x)  # (B, C, T', H', W')
        feat_vec = F.adaptive_avg_pool3d(feat, 1).flatten(1)  # (B, C)

        logits = self.model.head(self.model.norm(feat.permute(0, 4, 2, 3, 1)).permute(0, 4, 1, 2, 3))
        # 上面的 head 调用较复杂，直接复用原 forward 逻辑更稳
        logits = self.model(x)
        return {"logits": logits, "feat_list": [feat_vec]}


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
        self.head_in_features = self.model.head[1].in_features
        self.model.head[1] = nn.Linear(self.head_in_features, num_classes)
        self.num_classes = num_classes
        self.num_modal = num_modal

    def forward(self, x, alpha=1.0):
        # x: (B, T, 3, H, W) → (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # MViT 的 features 输出经过 encoder + layernorm 的 token 序列
        # 形状通常为 (B, num_tokens, embed_dim)
        feat = self.model.features(x)  # (B, N, C)
        feat_vec = feat.mean(dim=1)  # (B, C) 全局平均

        logits = self.model(x)
        modal_logits = torch.zeros((x.size(0), self.num_modal), device=x.device)
        return {"logits": logits, "feat_list": [feat_vec], "modal_logits": modal_logits}


# ============ 工厂函数 ============

def build_teacher(name: str, num_classes: int = 5):
    if name == "slowfast":
        return SlowFastTeacher(num_classes)
    if name == "video_swin":
        return VideoSwinTeacher(num_classes=num_classes)
    if name == "mvit":
        return MViTTeacher(num_classes=num_classes)
    raise ValueError(f"Unknown teacher: {name}")
