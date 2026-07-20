"""时空 Adapter：在 2D backbone 特征上轻量注入时空依赖

关键创新点 2：时空特征解耦 Adapter
- 3×3×3 浅 3D 卷积（depthwise）→ 时空局部
- 1D 时序卷积 → 时序依赖
- 残差连接
- 总参数 ~2M，NPU 友好
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSpatioTemporalConv(nn.Module):
    """3×3×3 depthwise 时空卷积

    输入: (B, C, T, H, W)
    输出: (B, C_out, T, H, W)

    设计：
    - 空间维度 depthwise（每个通道独立卷积）
    - 时序维度 pointwise（1D 卷积）
    - 比 3D 全卷积参数少 9× 左右
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        # 空间 depthwise: 每通道独立 3×3 卷
        self.spatial_dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
        # 时序 1D: 跨通道混合
        self.temporal_pw = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # 空间卷积：把 T 合并进 batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        x = self.spatial_dw(x)  # (B*T, C, H, W)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        # 时序 1D：把 H*W 合并进 batch
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, C, T)
        x = x.view(B * H * W, C, T)
        x = self.temporal_pw(x)  # (B*H*W, C_out, T)
        x = x.view(B, H, W, -1, T).permute(0, 3, 4, 1, 2).contiguous()  # (B, C_out, T, H, W)
        x = self.bn(x)
        x = self.act(x)
        return x


class TemporalConv1D(nn.Module):
    """1D 时序卷积块"""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.act(self.bn(self.conv(x)))


class SpatioTemporalAdapter(nn.Module):
    """时空 Adapter：在 2D backbone 特征上注入时空依赖

    输入: backbone 输出的单帧特征 (B, T, C, H, W)
    输出: (B, C_out, T, H, W) → GAP → (B, C_out, T)

    结构：
        3×3×3 DW ST Conv (C_in → C_mid)
            ↓
        1D Temporal Conv (C_mid → C_mid, k=5)
            ↓
        3×3×3 DW ST Conv (C_mid → C_out)
            ↓
        残差连接（C_in → C_out 经 1×1×1 conv 对齐）
    """

    def __init__(self, c_in: int = 128, c_mid: int = 64, c_out: int = 32,
                 temporal_kernel: int = 5):
        super().__init__()
        self.st_conv1 = DepthwiseSpatioTemporalConv(c_in, c_mid)
        self.temporal = TemporalConv1D(c_mid, c_mid, kernel_size=temporal_kernel)
        self.st_conv2 = DepthwiseSpatioTemporalConv(c_mid, c_out)
        # 残差对齐
        self.residual_align = nn.Conv3d(c_in, c_out, 1) \
            if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C_in, H, W) 来自 backbone（每帧独立）
        return: (B, C_out, T) GAP 后的时序特征
        """
        # 转为 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        residual = self.residual_align(x)
        out = self.st_conv1(x)        # (B, C_mid, T, H, W)
        # 1D 时序：把 H*W 合并
        B, C, T, H, W = out.shape
        out = out.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C, T)
        out = self.temporal(out)
        out = out.view(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()
        out = self.st_conv2(out)     # (B, C_out, T, H, W)
        out = out + residual         # 残差
        # GAP over H, W
        out = out.mean(dim=[3, 4])   # (B, C_out, T)
        return out


class CNN1DClassifier(nn.Module):
    """1D-CNN 时序分类头（NPU 友好，避免 Transformer 自注意力）

    输入: (B, C_out, T)
    输出: (B, num_classes)
    """

    def __init__(self, c_in: int = 32, c_hidden: int = 64, c_out: int = 128,
                 num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c_hidden)
        self.conv2 = nn.Conv1d(c_hidden, c_out, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.fc(x)                # (B, num_classes)
        return x
