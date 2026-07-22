"""时空 Adapter v3：6 层 STResBlock + 残差 + SE 通道注意力

关键创新点 2：时空特征解耦 Adapter（v3 升级版）
- 6 层 ST Conv（v2: 4 层）
- 残差连接（每 2 层一个残差）
- SE 通道注意力（v3 新增，提升特征表达）
- depthwise 降参数 + 加深
- 总参数 ~1.68M（v2: ~1.5M）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSpatioTemporalConv(nn.Module):
    """3×3×3 depthwise 时空卷积（v3 增强：带 SE 注意力）

    输入: (B, C, T, H, W)
    输出: (B, C_out, T, H, W)

    设计：
    - 空间 depthwise（每通道独立卷积）
    - 时序 pointwise（1D 卷积混合通道）
    - SE 通道注意力（v3 新增，提升特征表达）
    - 比 3D 全卷积参数少 9× 左右
    """

    def __init__(self, c_in: int, c_out: int, use_se: bool = True):
        super().__init__()
        self.spatial_dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
        self.temporal_pw = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU(inplace=True)
        # SE 通道注意力（v3 新增）
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(c_out, c_out // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(c_out // 4, c_out, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # 空间卷积：T 合并进 batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        x = self.spatial_dw(x)  # (B*T, C, H, W)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        # 时序 1D：H*W 合并进 batch
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, C, T)
        x = x.reshape(B * H * W, C, T)
        x = self.temporal_pw(x)  # (B*H*W, C_out, T)
        x = x.view(B, H, W, -1, T).permute(0, 3, 4, 1, 2).contiguous()  # (B, C_out, T, H, W)
        x = self.bn(x)
        x = self.act(x)
        if self.use_se:
            x = x * self.se(x)
        return x


class STResBlock(nn.Module):
    """时空残差块：2 层 ST Conv + 残差连接"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = DepthwiseSpatioTemporalConv(c_in, c_out)
        self.conv2 = DepthwiseSpatioTemporalConv(c_out, c_out)
        self.residual = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.bn_res = nn.BatchNorm3d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, H, W)
        residual = self.bn_res(self.residual(x))
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class TemporalConv1D(nn.Module):
    """1D 时序卷积块"""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpatioTemporalAdapterV3(nn.Module):
    """时空 Adapter v3：6 层 STResBlock + SE 通道注意力 + 残差

    输入: backbone 输出的单帧特征 (B, T, C, H, W)
    输出: (B, C_out, T) GAP 后的时序特征

    结构（depth=6 = 3 个 STResBlock，每个含 2 层 ST Conv）：
        [STResBlock 1] C_in → C_mid
            ↓
        [STResBlock 2..] C_mid → C_mid（中间 (depth-2)/2 个）
            ↓
        [STResBlock last] C_mid → C_out
            ↓
        GAP over H, W → (B, C_out, T)
            ↓
        1D 时序卷积（注入更长时序依赖）
    """

    def __init__(self, c_in: int = 256, c_mid: int = 256, c_out: int = 128,
                 temporal_kernel: int = 5, depth: int = 6,
                 residual: bool = True):
        super().__init__()
        assert depth % 2 == 0, "depth must be even (use STResBlock pairs)"
        self.depth = depth
        self.residual = residual

        # 第一个 STResBlock: C_in → C_mid
        self.block1 = STResBlock(c_in, c_mid)
        # 中间若干 STResBlock: C_mid → C_mid
        self.mid_blocks = nn.ModuleList([
            STResBlock(c_mid, c_mid) for _ in range((depth - 2) // 2)
        ])
        # 最后一个 STResBlock: C_mid → C_out
        self.block_last = STResBlock(c_mid, c_out)

        # 时序 1D 卷积（在 GAP 之前注入更长时序依赖）
        self.temporal = TemporalConv1D(c_out, c_out, kernel_size=temporal_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C_in, H, W) 来自 backbone（每帧独立）
        return: (B, C_out, T) GAP 后的时序特征
        """
        # 转为 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Block 1
        out = self.block1(x)        # (B, C_mid, T, H, W)
        # Mid blocks
        for blk in self.mid_blocks:
            out = blk(out)
        # Last block
        out = self.block_last(out)  # (B, C_out, T, H, W)
        # GAP over H, W
        out = out.mean(dim=[3, 4])   # (B, C_out, T)
        # 时序 1D 卷积
        out = self.temporal(out)     # (B, C_out, T)
        return out


# 向后兼容
SpatioTemporalAdapter = SpatioTemporalAdapterV3


class CNN1DClassifier(nn.Module):
    """1D-CNN 时序分类头 v3：支持 2-4 层 + 残差（NPU 友好）

    输入: (B, C, T)
    输出: (B, num_classes)
    """

    def __init__(self, c_in: int = 128, c_hidden: int = 512, c_out: int = 1024,
                 num_classes: int = 5, layers: int = 4,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        self.layers = layers

        # 4 层结构：c_in → c_hidden → c_hidden → c_hidden → c_out
        self.conv1 = nn.Conv1d(c_in, c_hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c_hidden)
        self.conv2 = nn.Conv1d(c_hidden, c_hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c_hidden)
        self.conv3 = nn.Conv1d(c_hidden, c_hidden, 3, padding=1) if layers >= 3 else None
        self.bn3 = nn.BatchNorm1d(c_hidden) if layers >= 3 else None
        self.conv4 = nn.Conv1d(c_hidden, c_out, 3, padding=1) if layers >= 4 else None
        self.bn4 = nn.BatchNorm1d(c_out) if layers >= 4 else None
        # 残差对齐
        self.res_align = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                         else nn.Identity())
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        res = self.res_align(x)        # (B, C_out, T)
        h = self.act(self.bn1(self.conv1(x)))   # (B, C_hidden, T)
        h = self.act(self.bn2(self.conv2(h)))   # (B, C_hidden, T)
        if self.layers >= 3 and self.conv3 is not None:
            h = self.act(self.bn3(self.conv3(h)))  # (B, C_hidden, T)
        if self.layers >= 4 and self.conv4 is not None:
            h = self.act(self.bn4(self.conv4(h)))   # (B, C_out, T)
            if self.residual:
                h = h + res                  # 残差相加
        x = self.pool(h).squeeze(-1)  # (B, C_out)
        x = self.fc(x)
        return x


class PoseAuxHead(nn.Module):
    """姿态辅助分支（关键创新）

    从 Adapter 特征预测 17 个 COCO 关键点 (x, y)
    训练期使用，推理期不部署

    功能：
    1. 让 backbone 学到人体结构特征（缓解 YOLO 特征不适配）
    2. 教师若有姿态输出，可做姿态蒸馏
    3. 输入特征 -> 17×2 坐标

    输入: (B, C, T) Adapter 输出
    输出: (B, T, 17, 2) 关键点坐标
    """

    def __init__(self, c_in: int = 64, num_kp: int = 17):
        super().__init__()
        self.conv = nn.Conv1d(c_in, 128, 3, padding=1)
        self.bn = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128, num_kp * 2)
        self.num_kp = num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.act(self.bn(self.conv(x)))     # (B, 128, T)
        x = x.transpose(1, 2)                   # (B, T, 128)
        x = self.fc(x)                          # (B, T, 17*2)
        x = x.view(x.size(0), x.size(1), self.num_kp, 2)
        return x
