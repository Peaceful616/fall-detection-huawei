"""全局配置"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ============ 模型 ============
    num_classes: int = 5  # Fall / Fall-like / ADL / Lying / Transition
    input_size: Tuple[int, int] = (320, 320)  # 算法输入分辨率
    seq_len: int = 16  # 时序窗口长度
    # YOLOv8n backbone（冻结）
    backbone_name: str = "yolov8n"
    backbone_freeze: bool = True
    backbone_channels: dict = field(default_factory=lambda: {
        "P3": 64, "P4": 128, "P5": 256  # YOLOv8n 三尺度输出通道
    })
    feat_stride: int = 16  # 时空 Adapter 取 P4 (stride=16)
    # 时空 Adapter
    adapter_c_in: int = 128
    adapter_c_mid: int = 64
    adapter_c_out: int = 32
    adapter_temporal_kernel: int = 5
    # 1D-CNN 分类头
    head_c_hidden: int = 64
    head_c_out: int = 128

    # ============ 蒸馏 ============
    distill_temperature: float = 4.0
    alpha_feat: float = 0.5   # 特征级
    alpha_logit: float = 1.0  # logit 级
    alpha_rkd: float = 0.2    # 关系级
    alpha_modal: float = 0.3  # 跨模态
    teacher_weights: dict = field(default_factory=lambda: {
        "slowfast": 0.5, "video_swin": 0.5
    })

    # ============ 训练 ============
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # ============ 数据 ============
    data_root: str = "./data/omnifall"
    ir_synthesis_prob: float = 0.3  # 红外合成概率
    num_workers: int = 4

    # ============ 长尾增强 ============
    aug_low_light_prob: float = 0.2
    aug_occlusion_prob: float = 0.3
    aug_perspective_prob: float = 0.2
    aug_motion_blur_prob: float = 0.2

    # ============ 部署 ============
    quantize_bits: int = 8
    export_platform: str = "rk3588"  # rk3588 / hisi


cfg = Config()
