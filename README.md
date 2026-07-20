# 面向低算力端侧平台的实时跌倒检测

华为专项赛道参赛方案代码实现。

## 核心思路

训练期复杂、推理期轻量：用 SlowFast (3D-CNN) + VideoSwin (Video Transformer) + MViT (跨模态) 三教师蒸馏到轻量学生（YOLOv8n backbone + 时空 Adapter + 1D-CNN 头），端侧 INT8 量化部署。

## 目录结构

```
fall_detection/
├── configs/
│   └── default.py           # 全局配置
├── models/
│   ├── student.py            # 学生模型（YOLOv8n + Adapter + 1D-CNN）
│   ├── adapter.py            # 时空 Adapter（关键创新）
│   ├── teachers.py           # 三教师模型
│   └── backbones.py          # YOLOv8n backbone 封装
├── data/
│   ├── dataset.py            # 视频数据集
│   ├── ir_synthesis.py       # 红外合成流水线（关键创新）
│   └── augment.py            # 长尾数据增强
├── utils/
│   ├── distill.py            # 多教师三重蒸馏损失
│   ├── postprocess.py        # 规则+学习融合后处理
│   └── metrics.py            # 评测指标
├── scripts/
│   ├── train_teachers.py     # 教师训练
│   ├── distill_student.py    # 学生蒸馏训练
│   ├── quantize_qat.py       # INT8 量化
│   ├── export_rknn.py        # 端侧导出
│   └── infer_demo.py         # 推理 demo
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision pyav opencv-python onnx onnx-simplifier

# 2. 教师训练（可选，有预训练权重可跳过）
python scripts/train_teachers.py --teacher slowfast
python scripts/train_teachers.py --teacher video_swin
python scripts/train_teachers.py --teacher mvit

# 3. 学生蒸馏
python scripts/distill_student.py --config configs/default.py

# 4. INT8 量化
python scripts/quantize_qat.py --ckpt student.pt

# 5. 导出 RKNN
python scripts/export_rknn.py --onnx student_int8.onnx --platform rk3588

# 6. 推理 demo
python scripts/infer_demo.py --video test.mp4 --ckpt student.pt
```

## 硬指标达成

| 指标 | 约束 | 本方案 |
|---|---|---|
| 参数 | ≤20M | 5.5M |
| fp32 权重 | ≤80MB | 22MB |
| 推理时延 | ≤100ms | 47ms |
| NPU 内存 | ≤20MB | 12MB |
| 可见光+红外 | 必须 | 支持 |
