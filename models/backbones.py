"""YOLOv8 backbone 封装：支持 YOLOv8n/s + 部分解冻末段

YOLOv8n backbone：~1.27M 参数（COCO 检测预训练）
YOLOv8s backbone：~9.4M 参数（COCO 检测预训练）

YOLOv8s 结构（前 10 个模块为 backbone）：
  0: Conv (stride 2)              3 → 32
  1: Conv (stride 4)              32 → 64
  2: C2f                          64 → 64
  3: Conv (stride 8)              64 → 128    ← P3 输出 (128 通道)
  4: C2f                          128 → 128
  5: Conv (stride 16)             128 → 256   ← P4 输出 (256 通道)
  6: C2f                          256 → 256
  7: Conv (stride 32)             256 → 512
  8: C2f                          512 → 512
  9: SPPF                         512 → 512   ← P5 输出 (512 通道)
"""
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class YOLOv8Backbone(nn.Module):
    """YOLOv8 backbone 封装，支持 YOLOv8n/s/m 等

    输入: (B, T, 3, H, W) -> reshape (B*T, 3, H, W) 逐帧提取
    输出: dict{
        'P3': (B*T, C3, H/8,  W/8),
        'P4': (B*T, C4, H/16, W/16),
        'P5': (B*T, C5, H/32, W/32),
    }
    """

    BACKBONE_END = 10  # index 0..9 是 backbone

    def __init__(self, name: str = "yolov8s", pretrained: bool = True,
                 freeze: bool = False, unfreeze_from: int = 7):
        """
        name: yolov8n / yolov8s / yolov8m 等
        freeze: True 全冻结；False 按 unfreeze_from 部分解冻末段
        unfreeze_from: 从该 index 起解冻（默认 7，解冻 Conv+C2f+SPPF 末段）
        """
        super().__init__()
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics not installed, run: pip install ultralytics"
            )
        weight = f"{name}.pt" if pretrained else f"{name}.yaml"
        model = YOLO(weight)
        # 取前 10 个模块（backbone，不含 neck/detect head）
        seq = model.model.model[:self.BACKBONE_END]
        self.backbone = nn.Sequential(*list(seq))

        # 冻结策略
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            # 部分解冻：unfreeze_from 之前的模块冻结，之后解冻
            modules = list(self.backbone)
            for i, m in enumerate(modules):
                requires_grad = (i >= unfreeze_from)
                for p in m.parameters():
                    p.requires_grad = requires_grad
                if not requires_grad:
                    m.eval()
            n_trainable = sum(p.numel() for p in self.backbone.parameters()
                             if p.requires_grad)
            n_total = sum(p.numel() for p in self.backbone.parameters())
            print(f"[Backbone {name}] trainable={n_trainable/1e6:.2f}M "
                  f"total={n_total/1e6:.2f}M (unfreeze from idx {unfreeze_from})")

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, 3, H, W)
        y = x
        P3 = P4 = P5 = None
        for i, m in enumerate(self.backbone):
            # 解冻的模块走 train 模式，冻结的走 eval（影响 BN）
            if m.training and not any(p.requires_grad for p in m.parameters()):
                m.eval()
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

    通过 c3/c4/c5 参数模拟 YOLOv8s 的通道数
    """

    def __init__(self, channels: dict, freeze: bool = True, unfreeze_from: int = 7):
        super().__init__()
        self.channels = channels
        c3, c4, c5 = channels["P3"], channels["P4"], channels["P5"]
        self.conv1 = nn.Conv2d(3, c3 // 4, 3, 2, 1)
        self.conv2 = nn.Conv2d(c3 // 4, c3 // 2, 3, 2, 1)
        self.conv3 = nn.Conv2d(c3 // 2, c3, 3, 2, 1)
        self.conv4 = nn.Conv2d(c3, c4, 3, 2, 1)
        self.conv5 = nn.Conv2d(c4, c5, 3, 2, 1)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        c1 = self.conv1(x)   # stride 2
        c2 = self.conv2(c1)  # stride 4
        c3 = self.conv3(c2)  # stride 8 (P3)
        c4 = self.conv4(c3)  # stride 16 (P4)
        c5 = nn.MaxPool2d(3, 1, 1)(self.conv5(c4))  # stride 32 (P5)
        return {"P3": c3, "P4": c4, "P5": c5}


def build_backbone(cfg):
    """根据环境自动选择真实 YOLOv8 或 dummy 占位"""
    if _ULTRALYTICS_AVAILABLE:
        try:
            return YOLOv8Backbone(
                name=cfg.backbone_name,
                pretrained=True,
                freeze=cfg.backbone_freeze,
                unfreeze_from=cfg.backbone_unfreeze_from,
            )
        except Exception as e:
            print(f"[WARN] YOLOv8 加载失败，回退到 DummyBackbone: {e}")
            return DummyBackbone(cfg.backbone_channels,
                                cfg.backbone_freeze,
                                cfg.backbone_unfreeze_from)
    return DummyBackbone(cfg.backbone_channels,
                        cfg.backbone_freeze,
                        cfg.backbone_unfreeze_from)
