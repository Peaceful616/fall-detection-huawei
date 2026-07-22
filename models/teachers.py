"""三教师模型：SlowFast / VideoSwin / MViT

教师只在训练期使用，推理时仅部署学生网络。
为兼容不同安装环境，提供轻量占位实现，真实训练时建议用官方预训练权重。
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
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Slow pathway: 每隔 4 帧取一帧
        slow = x[:, :, ::4, :, :]
        # Fast pathway: 原始帧率
        fast = x
        # 官方 SlowFast 输入格式是 list [slow, fast]
        logits = self.model([slow, fast])
        # 提取 logits 作为特征蒸馏用（后续可扩展为多层特征）
        return {"logits": logits, "feat_list": [logits]}


class SlowPathway(nn.Module):
    """SlowPathway：低帧率、大通道（占位实现，用于兼容）"""

    def __init__(self, c_in=3, c_out=64):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=(1, 7, 7),
                              stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(
            nn.Conv3d(c_out, 128, 3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.layer(x)


class FastPathway(nn.Module):
    """FastPathway：高帧率、小通道（占位实现，用于兼容）"""

    def __init__(self, c_in=3, c_out=8):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=(5, 7, 7),
                              stride=(1, 2, 2), padding=(2, 3, 3))
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.layer = nn.Sequential(
            nn.Conv3d(c_out, 32, 3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.layer(x)


# ============ VideoSwin 教师（Video Transformer 简化版）============

class TemporalSelfAttention(nn.Module):
    """简化的时序自注意力（仅时序维度）"""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class VideoSwinBlock(nn.Module):
    """简化版的 VideoSwin block"""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VideoSwinTeacher(nn.Module):
    """VideoSwin 教师模型简化实现

    输入: (B, T, 3, H, W)
    输出: logits + feat_list
    """

    def __init__(self, embed_dim: int = 96, num_classes: int = 5,
                 depths=(2, 2, 6, 2)):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 4, 4)
        self.blocks = nn.ModuleList([
            VideoSwinBlock(embed_dim, heads=4) for _ in range(sum(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])  # (B*T, 3, H, W)
        x = self.patch_embed(x)          # (B*T, C, H/4, W/4)
        # 全局平均池化到帧级特征
        x = x.mean(dim=[2, 3])          # (B*T, C)
        x = x.reshape(B, T, -1)          # (B, T, C) - 帧级 token
        # 时序自注意力（token 数 = T，可控）
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, C)
        return {"logits": self.fc(x), "feat_list": [x]}


# ============ MViT 跨模态教师 ============

class GradientReversalFn(torch.autograd.Function):
    """梯度反转层（用于模态对抗）"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class MViTTeacher(nn.Module):
    """MViT 跨模态教师：RGB + 红外模态不变特征学习

    主任务: 5 分类
    模态对抗: 让主干学习模态不变特征

    输入: (B, T, 3, H, W) - RGB 或红外
    输出:
        logits: (B, num_classes)
        feat_list: List[Tensor]
        modal_logits: (B, 2) - 模态分类（仅训练用）
    """

    def __init__(self, embed_dim: int = 96, num_classes: int = 5,
                 num_modal: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, embed_dim, 4, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        # 模态对抗头
        self.modal_head = nn.Linear(embed_dim, num_modal)
        self.lambda_adv = 0.1

    def forward(self, x, alpha=1.0):
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        x = self.conv(x)
        x = x.flatten(1)  # (B*T, C*H*W)
        feat = x.view(B, T, -1).mean(1)  # (B, C*H*W)
        feat = feat[:, :self.conv.out_channels]  # 取前 C 维
        # 主任务
        feat_adv = GradientReversalFn.apply(feat, self.lambda_adv * alpha)
        modal_logits = self.modal_head(feat_adv)
        logits = self.fc(feat)
        return {"logits": logits, "feat_list": [feat], "modal_logits": modal_logits}


# ============ 工厂函数 ============

def build_teacher(name: str, num_classes: int = 5):
    if name == "slowfast":
        return SlowFastTeacher(num_classes)
    if name == "video_swin":
        return VideoSwinTeacher(num_classes=num_classes)
    if name == "mvit":
        return MViTTeacher(num_classes=num_classes)
    raise ValueError(f"Unknown teacher: {name}")
