"""全局配置 - v3 升级版

v3 升级要点（充分利用 20MB 参数预算，目标 ~12M）：
- backbone YOLOv8s 部分解冻 → 全解冻（特征深度适配跌倒任务）
- Adapter v3：4 层 → 6 层，c_mid 128→256, c_out 64→128
- head v3：c_hidden 128→256, c_out 256→512
- 新增多层特征蒸馏（P3+P4+P5 三层对齐 SlowFast 多 stage）
- 蒸馏温度 T=6 → T=8（教师能力进一步增强）
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ============ 模型 ============
    num_classes: int = 5  # Fall / Fall-like / ADL / Lying / Transition
    input_size: Tuple[int, int] = (288, 288)  # 算法输入分辨率（省内存）
    seq_len: int = 16  # 时序窗口长度

    # YOLOv8s backbone（v3: 全解冻）
    backbone_name: str = "yolov8s"
    backbone_freeze: bool = False
    backbone_unfreeze_from: int = 0  # v3: 0 = 全解冻
    backbone_channels: dict = field(default_factory=lambda: {
        "P3": 128, "P4": 256, "P5": 512  # YOLOv8s 三尺度输出通道
    })
    feat_stride: int = 16  # 时空 Adapter 取 P4 (stride=16)

    # 时空 Adapter v3（扩容 + 加深）
    adapter_c_in: int = 256         # P4 通道
    adapter_c_mid: int = 256        # v2:128 → v3:256
    adapter_c_out: int = 128        # v2:64  → v3:128
    adapter_temporal_kernel: int = 5
    adapter_depth: int = 6          # v2:4 → v3:6
    adapter_residual: bool = True

    # 1D-CNN 分类头 v3
    head_c_hidden: int = 512        # v2:256 → v3:512
    head_c_out: int = 1024          # v2:512 → v3:1024
    head_layers: int = 4            # v3: 3→4
    head_residual: bool = True

    # 姿态辅助分支（关键创新，训练期使用）
    aux_kp_enabled: bool = True
    aux_kp_num: int = 17            # COCO 17 关键点
    aux_kp_weight: float = 0.3

    # 多层特征蒸馏（v3 新增）
    multi_level_feat_distill: bool = True  # P3+P4+P5 三层对齐
    alpha_feat_multilevel: float = 0.3

    # ============ 蒸馏 ============
    distill_temperature: float = 8.0  # v2:6 → v3:8
    alpha_feat: float = 0.5
    alpha_logit: float = 1.0
    alpha_rkd: float = 0.2
    alpha_modal: float = 0.3
    alpha_aux: float = 0.3
    teacher_weights: dict = field(default_factory=lambda: {
        "slowfast": 0.4, "video_swin": 0.4, "mvit": 0.2
    })

    # ============ 训练 ============
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    lr_backbone: float = 1e-5      # backbone 全解冻用更小学习率
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # ============ 数据 ============
    data_root: str = "./data/omnifall"
    ir_synthesis_prob: float = 0.3
    num_workers: int = 4

    # ============ 长尾增强 ============
    aug_low_light_prob: float = 0.2
    aug_occlusion_prob: float = 0.3
    aug_perspective_prob: float = 0.2
    aug_motion_blur_prob: float = 0.2

    # ============ 部署 ============
    quantize_bits: int = 8
    export_platform: str = "rk3588"


cfg = Config()

