"""全局配置 - v2 升级版

升级要点：
- backbone YOLOv8n → YOLOv8s（解冻末段，特征适配跌倒任务）
- Adapter 扩容 + 加深（2 层 → 4 层，残差连接）
- 1D-CNN head 扩容 + 残差
- 新增姿态辅助分支（训练期使用，推理期不部署）
- 输入分辨率 320×320 → 288×288（省 NPU 内存）
- 蒸馏温度 T=4 → T=6（教师能力增强需调大温度）
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ============ 模型 ============
    num_classes: int = 5  # Fall / Fall-like / ADL / Lying / Transition
    input_size: Tuple[int, int] = (288, 288)  # 算法输入分辨率（省内存）
    seq_len: int = 16  # 时序窗口长度

    # YOLOv8s backbone（部分解冻末段）
    backbone_name: str = "yolov8s"  # yolov8n → yolov8s（参数 9.4M）
    backbone_freeze: bool = False  # 改为 False，按 backbone_unfreeze_from 解冻末段
    backbone_unfreeze_from: int = 7  # 从 index 7 起解冻（C2f + SPPF）
    backbone_channels: dict = field(default_factory=lambda: {
        "P3": 128, "P4": 256, "P5": 512  # YOLOv8s 三尺度输出通道
    })
    feat_stride: int = 16  # 时空 Adapter 取 P4 (stride=16)

    # 时空 Adapter（扩容 + 加深）
    adapter_c_in: int = 256         # P4 通道（YOLOv8s 比 YOLOv8n 翻倍）
    adapter_c_mid: int = 128        # 64 → 128
    adapter_c_out: int = 64         # 32 → 64
    adapter_temporal_kernel: int = 5
    adapter_depth: int = 4          # 新增：4 层 ST Conv
    adapter_residual: bool = True   # 新增：残差连接

    # 1D-CNN 分类头（扩容 + 加深 + 残差）
    head_c_hidden: int = 128        # 64 → 128
    head_c_out: int = 256           # 128 → 256
    head_layers: int = 3            # 2 → 3
    head_residual: bool = True

    # 姿态辅助分支（关键创新，训练期使用）
    aux_kp_enabled: bool = True
    aux_kp_num: int = 17            # COCO 17 关键点
    aux_kp_weight: float = 0.3      # 辅助损失权重

    # ============ 蒸馏 ============
    distill_temperature: float = 6.0  # 4.0 → 6.0（教师变强需调大温度）
    alpha_feat: float = 0.5
    alpha_logit: float = 1.0
    alpha_rkd: float = 0.2
    alpha_modal: float = 0.3
    alpha_aux: float = 0.3          # 姿态辅助蒸馏权重
    teacher_weights: dict = field(default_factory=lambda: {
        "slowfast": 0.4, "video_swin": 0.4, "mvit": 0.2
    })

    # ============ 训练 ============
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    lr_backbone: float = 1e-5      # backbone 末段解冻用更小学习率
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
