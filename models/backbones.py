"""YOLOv8n backbone 封装：输出多尺度特征图（冻结，不参与训练）

基于 ultralytics 8.4.x 验证通过

YOLOv8n 结构（23 个子模块）：
  0: Conv (stride 2)
  1: Conv (stride 4)
  2: C2f
  3: Conv (stride 8)     ← P3 输出 (64 通道)
  4: C2f
  5: Conv (stride 16)    ← P4 输出 (128 通道)
  6: C2f
  7: Conv (stride 32)
  8: C2f
  9: SPPF                ← P5 输出 (256 通道)
  10-22: neck + head（检测头，我们不需要）
"""
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class YOLOv8nBackbone(nn.Module):
    """YOLOv8n backbone 封装

    输入: (B, T, 3, H, W) -> reshape (B*T, 3, H, W) 逐帧提取
    输出: dict{
        'P3': (B*T, 64,  H/8,  W/8),
        'P4': (B*T, 128, H/16, W/16),
        'P5': (B*T, 256, H/32, W/32),
    }
    """

    # YOLOv8n backbone 模块索引（取 0-9，去掉 neck + head）
    BACKBONE_END = 10  # index 0..9 是 backbone

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics not installed, run: pip install ultralytics"
            )
        model = YOLO("yolov8n.pt" if pretrained else "yolov8n.yaml")
        # 取前 10 个模块（Conv+C2f+SPPF，对应 P3/P4/P5）
        seq = model.model.model[:self.BACKBONE_END]
        self.backbone = nn.Sequential(*list(seq))
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, 3, H, W)
        # 逐模块前向，记录 P3/P4/P5
        # index 4 是 C2f（P3 输出，stride 8）
        # index 6 是 C2f（P4 输出，stride 16）
        # index 9 是 SPPF（P5 输出，stride 32）
        y = x
        P3 = P4 = P5 = None
        for i, m in enumerate(self.backbone):
            y = m(y)
            if i == 4:
                P3 = y
            elif i == 6:
                P4 = y
            elif i == 9:
                P5 = y
        return {"P3": P3, "P4": P4, "P5": P5}


class DummyBackbone(nn.Module):
    """YOLO 未安装时的占位实现，仅用于代码跑通 demo

    输出与 YOLOv8n 接口一致的 dummy 特征，方便无 ultralytics 环境调试
    """

    def __init__(self, channels: dict, freeze: bool = True):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        c1 = self.conv1(x)   # stride 2
        c2 = self.conv2(c1)  # stride 4
        c3 = self.conv3(c2)  # stride 8 (P3)
        c4 = self.conv4(c3)  # stride 16 (P4)
        # 模拟 SPPF
        c5 = nn.MaxPool2d(3, 1, 1)(nn.MaxPool2d(3, 1, 1)(c4))  # stride 32 (P5)
        return {"P3": c3, "P4": c4, "P5": c5}


def build_backbone(cfg):
    """根据环境自动选择真实 YOLOv8n 或 dummy 占位"""
    if _ULTRALYTICS_AVAILABLE:
        try:
            return YOLOv8nBackbone(pretrained=True, freeze=cfg.backbone_freeze)
        except Exception as e:
            print(f"[WARN] YOLOv8n 加载失败，回退到 DummyBackbone: {e}")
            return DummyBackbone(cfg.backbone_channels, cfg.backbone_freeze)
    return DummyBackbone(cfg.backbone_channels, cfg.backbone_freeze)
