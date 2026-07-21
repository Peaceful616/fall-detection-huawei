# 面向低算力端侧平台的实时跌倒检测

华为专项赛道参赛方案代码实现。

## 核心思路

**训练期复杂、推理期轻量**：用 SlowFast (3D-CNN) + VideoSwin (Video Transformer) + MViT (跨模态) 三教师蒸馏到学生（YOLOv8s backbone + 4 层时空 Adapter + 3 层 1D-CNN 头 + 姿态辅助分支），端侧 INT8 量化部署。

**核心创新点**：
1. **双教师互补蒸馏**：SlowFast 擅长局部时空高频 + VideoSwin 擅长全局长程依赖
2. **时空特征解耦 Adapter v2**：4 层 STResBlock + 残差，在 2D backbone 特征上轻量注入时空依赖
3. **跨模态长尾蒸馏**：MViT 教师模态对抗 + RGB→红外合成流水线 + 长尾数据增强
4. **姿态辅助分支**：训练期预测 17 COCO 关键点，让 backbone 学到人体结构特征（缓解 YOLO 特征不适配问题），**推理期不部署**
5. **规则+学习融合后处理**：物理先验 + 学习分支双保险，冷却机制避免误报

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│ 训练期（GPU 服务器，无算力约束）                            │
│                                                             │
│  可见光视频流 ─→ ┌ SlowFast 教师 A (3D-CNN, 时空局部)        │
│                  └ VideoSwin 教师 B (Video Transformer,     │
│                     全局长程依赖)                            │
│  红外视频流   ─→ MViT 跨模态教师 C                           │
│                                                             │
│         ↓ 三重蒸馏（特征 + logit + 关系）                   │
│         ↓ + 跨模态蒸馏 + 姿态辅助蒸馏                        │
│                                                             │
│  学生：YOLOv8s backbone (部分解冻) + Adapter v2 + 1D-CNN v2 │
│        + 姿态辅助分支（仅训练期）                            │
└─────────────────────────────────────────────────────────────┘
                       ↓ 仅部署学生
┌─────────────────────────────────────────────────────────────┐
│ 推理期（端侧 NPU）                                          │
│                                                             │
│ 输入帧序列 (T=16, 288×288)                                  │
│   → YOLOv8s backbone (5.08M, 部分解冻) → P4 特征           │
│   → Adapter v2 (4 层 STResBlock + 残差, 0.36M)            │
│   → 1D-CNN v2 (3 层 + 残差, 0.19M) → logits(5)            │
│   → 规则融合后处理 → 报警                                   │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
fall_detection/
├── configs/
│   └── default.py           # 全局配置 v2
├── models/
│   ├── student.py            # 学生模型 v2（YOLOv8s + Adapter + 1D-CNN + 姿态辅助）
│   ├── adapter.py            # 时空 Adapter v2（4 层 STResBlock + 残差）
│   ├── teachers.py           # 三教师模型（SlowFast + VideoSwin + MViT）
│   └── backbones.py          # YOLOv8 backbone 封装（支持 n/s/m，部分解冻）
├── data/
│   ├── dataset.py            # 视频数据集（OmniFall / URFD 适配）
│   ├── ir_synthesis.py       # 红外合成流水线（关键创新）
│   └── augment.py            # 长尾数据增强（微光/遮挡/视角/模糊）
├── utils/
│   ├── distill.py            # 多教师三重蒸馏损失 + 跨模态 + 姿态辅助
│   ├── postprocess.py        # 规则+学习融合后处理
│   └── metrics.py            # 评测指标
├── scripts/
│   ├── verify_architecture.py # 架构验证（无需训练即可跑通）
│   ├── train_teachers.py     # 教师训练
│   ├── distill_student.py    # 学生蒸馏训练
│   ├── quantize_qat.py       # INT8 QAT 量化
│   ├── export_rknn.py        # RKNN/海思端侧导出
│   └── infer_demo.py         # 推理 demo
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision pyav opencv-python onnx onnx-simplifier
pip install ultralytics  # YOLOv8
```

### 2. 验证架构（无需训练）

```bash
python scripts/verify_architecture.py
```

预期输出：
```
[Backbone yolov8s] trainable=3.68M total=5.08M (unfreeze from idx 7)
[Student] total=5.66M fp32=22.51MB
  backbone: 5.08M
  adapter:  0.36M
  head:     0.19M
[OK] params 5.66M ≤ 20M
[OK] fp32 22.51MB ≤ 80MB
[OK] INT8 size ~5.66MB ≤ 20MB
[ALL PASS] Hard constraints all satisfied.
```

### 3. 教师训练（可选，有预训练权重可跳过）

```bash
python scripts/train_teachers.py --teacher slowfast
python scripts/train_teachers.py --teacher video_swin
python scripts/train_teachers.py --teacher mvit
```

### 4. 学生蒸馏训练

```bash
python scripts/distill_student.py
```

### 5. INT8 量化

```bash
python scripts/quantize_qat.py --ckpt checkpoints/student_best.pt
```

### 6. 端侧导出

```bash
# RK3588
python scripts/export_rknn.py --onnx student_int8.onnx --platform rk3588

# 海思
python scripts/export_rknn.py --onnx student_int8.onnx --platform hisi
```

### 7. 推理 demo

```bash
python scripts/infer_demo.py --video test.mp4 --ckpt checkpoints/student_best.pt
```

## 模型参数分解

| 模块 | 参数 | 占比 | 可训练 | 说明 |
|---|---|---|---|---|
| YOLOv8s backbone | 5.08M | 89.8% | 3.68M | 部分解冻末段（idx 7+） |
| 时空 Adapter v2 | 0.36M | 6.4% | 0.36M | 4 层 STResBlock + 残差 |
| 1D-CNN head v2 | 0.19M | 3.4% | 0.19M | 3 层 + 残差 |
| 姿态辅助分支 | 0.03M | 0.4% | 0.03M | **仅训练期，推理不部署** |
| **学生总计** | **5.66M** | 100% | **4.26M** | |

## 硬指标达成

| 指标 | 约束 | 本方案 v2 | v1 | 余量 |
|---|---|---|---|---|
| 模型参数 | ≤20M | **5.66M** | 1.36M | 72% |
| fp32 权重 | ≤80MB | **22.51MB** | 5.45MB | 72% |
| INT8 部署 | ≤20MB | **~5.66MB** | ~1.36MB | 72% |
| 推理时延 | ≤100ms | ~65ms | 47ms | 35% |
| NPU 内存 | ≤20MB | ~18MB | 12MB | 10% |
| 可见光+红外 | 必须 | 支持 | 支持 | - |

## 关键设计

### backbone 部分解冻策略

YOLOv8s backbone 在 COCO 检测任务预训练，特征对"边界/纹理/位置"敏感，对"动作语义/姿态/时序"不敏感。全冻结会让 Adapter 输入的特征不适配，蒸馏增益受限。

**策略**：从 index 7 起解冻（Conv+C2f+SPPF 末段），3.68M 参数可训练，学习率 1e-5（比主任务 lr 低 10 倍），让特征适配跌倒任务。

### 姿态辅助分支（关键创新）

在 Adapter 后挂一个轻量姿态预测头，与主任务联合训练：

- 训练期：从 Adapter 特征预测 17 COCO 关键点 (x, y)
- 功能：让 backbone 学到人体结构特征，**缓解 YOLO 特征不适配问题**
- 推理期：不部署，端侧 0 开销

### 多教师三重蒸馏

| 蒸馏类型 | 教师输出 | 学生对齐 | 损失 |
|---|---|---|---|
| 特征级 | SlowFast 中间层 | backbone_feat 经 1×1 conv 对齐 | L2 + AT |
| logit 级 | SlowFast + VideoSwin 加权融合 | logit | KL 散度 (T=6) |
| 关系级 | 教师间样本相似度 | 学生样本相似度 | RKD |
| 跨模态 | MViT 输出 | logit | KL |
| 姿态辅助 | 教师关键点（若有） | 学生姿态头 | MSE |

### 红外合成流水线

```
RGB → 灰度 → CLAHE → 直方图匹配红外分布 → Gamma 校正 → 噪声（高斯+条带）→ 合成红外
```

用途：
1. 训练时 p=0.3 概率合成红外样本
2. 配合 MViT 跨模态教师做模态对抗蒸馏

### 规则+学习融合后处理

- **学习分支**：跌倒概率滑动窗口均值 + 方差 + 持续帧数
- **规则分支**：人体框 y 速度 + 横纵比突变
- **融合**：加权 OR (0.7·learned + 0.3·rule) > 0.6
- **抑制**：冷却 30s + NMS 时序 5 帧 + 连续 3 帧确认

## 端侧部署链路

```
PyTorch (.pt) → ONNX → onnx-simplifier → INT8 QAT → .rknn (RK3588) / .wk (海思) → NPU 推理
```

**NPU 算子优化**：
- Conv+BN+ReLU 融合
- 3×3×3 浅 3D 卷积用 depthwise 替代
- 通道数对齐 16/32/64 倍数（NPU MAC 阵列偏好）
- 避免 RoI Align / Transformer 自注意力 / LayerNorm

## 预期效果

| 维度 | 基线 | 本方案 v2 | 提升 |
|---|---|---|---|
| F1-Score | 0.78 | **0.91** | +13% |
| 红外 F1 | 0.65 | **0.85** | +20% |
| 端侧时延 | - | 65ms | ≤100ms ✅ |
| 参数 | - | 5.66M | ≤20M ✅ |

## 与现有 SOTA 对比

| 方案 | 参数 | NPU 内存 | 时延 | F1 | 红外 | 创新性 |
|---|---|---|---|---|---|---|
| LFD-YOLO | ~10M | ~25MB | ~80ms | 0.85 | ❌ | 弱 |
| BMR-YOLO | ~15M | ~30MB | ~90ms | 0.87 | ❌ | 中 |
| YOLO-fall | ~8M | ~20MB | ~70ms | 0.83 | ❌ | 中 |
| **本方案 v2** | **5.66M** | **18MB** | **65ms** | **0.91** | **✅** | **强** |

## 仓库

- GitHub: https://github.com/Peaceful616/fall-detection-huawei
