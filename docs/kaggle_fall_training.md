# Kaggle Fall Dataset Training Report

## Overview

This document records the first full training round of the three-teacher distillation pipeline on the Kaggle Fall Video Dataset. It validates that the proposed architecture (YOLOv8s student + SlowFast/VideoSwin/MViT teachers + multi-level distillation) is end-to-end trainable and reaches strong binary fall-detection metrics on the in-domain validation set.

**Training dates**: 2026-07-22 ~ 2026-07-23
**Server**: AutoDL container (vGPU 32GB, 12 vCPU, 62GB RAM)
**Dataset**: Kaggle Fall Video Dataset (binary: Fall / ADL)
**Round tag**: `kaggle_50ep`

---

## 1. Dataset

### 1.1 Source

Kaggle Fall Video Dataset (`https://www.kaggle.com/datasets/payutch/fall-video-dataset`). Raw structure:

```
archive/
├── Fall/
│   ├── Raw_Video/*.mp4
│   └── Keypoints_CSV/*_keypoints.csv   # 17 COCO keypoints per frame
└── No_Fall/
    ├── Raw_Video/*.mp4
    └── Keypoints_CSV/*_keypoints.csv
```

### 1.2 Adaptation

`data/kaggle_fall_adapter.py` converts the raw layout into the unified format consumed by `data/dataset.py`:

- Copies/symlinks videos to `data/kaggle_fall/videos/` with prefix `fall_` / `adl_`.
- Writes `annotations_train.json` / `annotations_val.json` (80/20 split, `random.seed(42)`, **split by whole video** to prevent leakage).
- Indexes keypoint CSVs (absolute path stored in `keypoint_csv` field) for the pose auxiliary branch.
- Label mapping: `Fall → 1`, `No_Fall → 0` (ADL). Classes 2/3/4 (Fall-like/Lying/Transition) are not present in this dataset.

```bash
python data/kaggle_fall_adapter.py ./data/kaggle_fall_raw ./data/kaggle_fall
```

### 1.3 Pre-decoding

`scripts/predecode_videos.py` uniformly samples 16 frames per video at 224×224 and saves as JPEG under `data/kaggle_fall/frames/<video_stem>/frame_0000.jpg ... frame_0015.jpg`. `VideoFallDatasetFast` then reads PIL images instead of decoding video, giving 5-10× I/O speedup.

```bash
python scripts/predecode_videos.py --data_root ./data/kaggle_fall --seq_len 16 --input_size 224
```

### 1.4 Statistics

| Split | Samples | Fall | ADL | With keypoints |
|---|---|---|---|---|
| Train | 5590 | — | — | subset |
| Val | 1398 | — | — | subset |

(Fall/ADL balance per split to be filled from `annotations_*.json` after first run.)

---

## 2. Model Architecture

### 2.1 Student

- **Backbone**: YOLOv8s (5.08M params, fully unfrozen from idx 0)
- **Temporal Adapter**: 6-layer 1D-CNN, `c_mid=256`, `c_out=128`, kernel=5, residual
- **Head**: 4-layer 1D-CNN, `c_hidden=512`, `c_out=1024`, residual
- **Auxiliary**: 17 COCO keypoints regression head (training-only, pose supervision)
- **Total**: 10.3M params, 41.0 MB fp32 — **within the 20M / 80MB hard limit** of the competition.

### 2.2 Teachers (frozen during distillation)

| Teacher | Source | Pretrained on | Role |
|---|---|---|---|
| SlowFast R50 | `pytorchvideo` | Kinetics-400 | 3D-CNN temporal features |
| Video Swin Transformer T | `torchvision` | Kinetics-400 | Video transformer features |
| MViT v2 S | `torchvision` | Kinetics-400 | Cross-modal logits + token features |

All three teachers had their classification heads replaced with `nn.Linear(in_features, 5)` and fine-tuned on Kaggle Fall before distillation. Their checkpoints are stored as `checkpoints/teacher_<name>_best.pt`.

### 2.3 Distillation Loss

`utils/distill.py::DistillLoss` combines six terms:

| Term | Weight | Aligns student → teacher |
|---|---|---|
| CE (main task) | 1.0 | — |
| Feature L2 + AT | `alpha_feat=0.5` | backbone GAP feat ↔ SlowFast feat |
| Logit KL (T=8) | `alpha_logit=1.0` | logits ↔ weighted 3-teacher soft labels |
| RKD (pairwise dist) | `alpha_rkd=0.2` | sample relations ↔ VideoSwin feat |
| Modal KL | `alpha_modal=0.3` | logits ↔ MViT logits |
| Aux pose MSE | `alpha_aux=0.3` | 17 keypoints ↔ GT keypoints (masked) |

Feature alignment uses `nn.Linear` adapters (`feat_align_slowfast`, `feat_align_video_swin`) created lazily on first forward.

---

## 3. Training

### 3.1 Config (`configs/default.py`)

| Item | Value |
|---|---|
| Epochs | 100 (saved best at epoch ~25-30) |
| Batch size | 4 |
| LR | 1e-4 (AdamW) |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR, T_max=epochs |
| Num workers | 8, pin_memory, prefetch_factor=2 |
| Input | (16, 3, 224, 224) |
| Temperature | 8.0 |
| AMP | enabled (mixed precision) |

### 3.2 Command

```bash
nohup python scripts/distill_student.py \
    --epochs 100 --batch_size 4 --lr 1e-4 \
    > logs/distill.log 2>&1 &
tail -f logs/distill.log
```

### 3.3 Loss curve (epoch 1 sample)

```
[Distill Epoch 1] total=8.6777  ce=0.5759  feat=0.0733  logit=5.1649  rkd=0.0618  modal=6.3014  aux=3.3246
```

- `logit` and `modal` dominate (expected — student is absorbing teacher knowledge).
- `ce=0.576` is well below `ln(5)=1.61`, indicating the backbone already carries useful features.
- `feat`/`rkd` are small — feature spaces are roughly aligned after adapters initialize.

### 3.4 Fixes applied during this round

1. **SlowFastTeacher forward** — `body_blocks` is an `nn.ModuleList` slice without a `forward`; calling it directly raised `NotImplementedError`. Fixed by iterating block-by-block (`models/teachers.py`).
2. **VideoSwinTeacher forward** — `self.model.features(x)` skipped `patch_embed`/`pos_drop`, causing `RuntimeError: normalized_shape=[96] vs input [8,3,16,224,224]`. Fixed by following the official `patch_embed → pos_drop → features → norm → permute → avgpool → flatten → head` order.
3. **MViTTeacher forward** — torchvision MViT has **no `self.features`** attribute (patch embed is `conv_proj`, encoder is `self.blocks`). Fixed with a forward hook on `self.model.norm` to capture the token sequence as the feature, and a head-layout guard for both `nn.Sequential` and `nn.Linear` head versions.
4. **DistillLoss feat-align check** — used `out_channels` (Conv attribute) on `nn.Linear` modules; `nn.Linear` exposes `out_features`. Fixed both checks (`utils/distill.py`).

All four fixes are committed on `main`:
`d5e0edc` → `b7e913f` → `a00fb6a`.

---

## 4. Results

### 4.1 Validation metrics (best, ~epoch 25-30)

```
===== Student Validation =====
Precision: 0.9968
Recall:    0.9984
F1-Score:  0.9976
TP=630 FP=2 FN=1
```

Saved to `checkpoints/student_best.pt` (F1=0.9976).

### 4.2 Interpretation (important)

This F1 is an **in-domain sanity check, not a generalization proof**:

- The validation set is the held-out 20% split of the same Kaggle dataset (controlled indoor scenes, stable background, standard fall motions). It does not represent real-world deployment conditions.
- Kaggle Fall is **binary** (Fall / ADL); the 5-class fine-grained distinction (Fall vs Fall-like vs Lying vs Transition) is not exercised.
- `VideoFallDatasetFast` samples the **first 16 frames** deterministically for both train and val, removing temporal randomness.
- The score only confirms that: **the three-teacher + student distillation pipeline trains end-to-end and reaches the upper bound on an easy in-domain benchmark.**

Generalization evidence must come from cross-domain evaluation on OmniFall (next round, see `docs/omnifall_integration.md`).

---

## 5. Artifact Archive

All checkpoints + logs from this round are archived via:

```bash
python scripts/archive_round.py --tag kaggle_50ep
```

Produces `checkpoints/round_kaggle_50ep/`:

```
student_best.pt
teacher_slowfast_best.pt
teacher_video_swin_best.pt
teacher_mvit_best.pt
distill.log
manifest.json          # epoch / f1 / sha256 / timestamp
```

`manifest.json` records `epoch`, `f1`, per-file `sha256` and size, so the round is reproducible and auditable.

---

## 6. Next Steps

1. **OmniFall integration** (`docs/omnifall_integration.md`) — replace Kaggle with `of-sta-cs` (8 staged datasets, 16-class taxonomy, cross-subject split) for multi-class training.
2. **Cross-domain evaluation** — train on `of-sta-cs`, test on `of-sta-to-all-cs` (staged + ItW + Syn) for real generalization numbers.
3. **Infrared support verification** — inspect OmniFall sub-datasets (UP-Fall / CMDFall are multi-modal) for infrared views.
4. **NPU export** — `scripts/export_rknn.py` to ONNX → RKNN, measure on-device latency on RK3588 (deferred to finals per resource planning).
