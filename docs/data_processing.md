# Data Processing

## Training Pipeline Optimization (2026-07-22)

### Problem

Training with `train_teachers.py` was extremely slow on the remote server (vGPU 32GB, 12 vCPU, 62GB RAM). Root cause analysis identified the following bottlenecks:

1. **Synchronous video I/O** — `VideoFallDataset.__getitem__` used `cv2.VideoCapture` to decode video frames on-the-fly, causing severe I/O blocking. GPU utilization was only 30-40%.
2. **Large input resolution** — 288x288 input with 16 frames per sample consumed excessive compute.
3. **No mixed precision** — FP32-only training underutilized the GPU.
4. **Suboptimal DataLoader** — Only 4 workers, no `pin_memory` or `prefetch_factor`.

### Changes

#### 1. Pre-decode Videos to Frames

**New script:** `scripts/predecode_videos.py`

Decodes all videos in `data_root/videos/` into per-video frame directories under `data_root/frames/`. Each video is uniformly sampled into `seq_len` frames and saved as JPEG at `input_size` resolution.

```bash
python scripts/predecode_videos.py --data_root ./data/kaggle_fall --seq_len 16 --input_size 224
```

#### 2. Fast Dataset Class

**Modified:** `data/dataset.py`

Added `VideoFallDatasetFast` — reads pre-decoded JPEG frames via PIL instead of decoding video with OpenCV. `build_datasets()` auto-detects the `frames/` directory and uses the fast class when available, falling back to `VideoFallDataset` otherwise.

#### 3. AMP Mixed Precision Training

**Modified:** `scripts/train_teachers.py`

Added `torch.cuda.amp.autocast` + `GradScaler` in the training loop. FP16 forward/backward reduces memory usage and increases throughput by 30-50%.

#### 4. DataLoader Optimization

**Modified:** `scripts/train_teachers.py` + `configs/default.py`

- `num_workers`: 4 → 8 (leveraging 12 vCPU)
- `pin_memory=True` for faster CPU→GPU transfer
- `prefetch_factor=2` for data prefetching
- `non_blocking=True` on `.to(device)` calls

#### 5. Input Resolution Reduction

**Modified:** `configs/default.py`

- `input_size`: (288, 288) → (224, 224)
- Computation reduced by ~45%; 224x224 is sufficient for fall detection tasks.
- Note: Camera input is 1080P+, resized to 224x224 at inference time.

### Expected Performance Improvement

| Optimization | Speedup |
|---|---|
| Pre-decoded frames (eliminate I/O blocking) | 5-10x |
| Resolution 224x224 | 1.5x |
| AMP mixed precision | 1.3-1.5x |
| DataLoader tuning (workers, pin_memory, prefetch) | 1.2x |
| **Combined** | **10-20x** |

GPU utilization expected to increase from 30-40% to 80-95%.

### Storage Estimate

Assuming ~1000 videos, 16 frames each at 224x224 JPEG (~30 KB/frame):
- Per video: ~480 KB
- Total: ~480 MB (well within 50 GB data disk)

### 6. SlowFast Teacher Model Upgrade

**Modified:** `models/teachers.py`

Replaced the hand-rolled `SlowFastTeacher` (1.17M params, randomly initialized) with the official PyTorchVideo SlowFast R50 pretrained on Kinetics 400:

```python
self.model = torch.hub.load(
    "facebookresearch/pytorchvideo:main",
    "slowfast_r50",
    pretrained=True,
)
# Replace final classification head
in_features = self.model.blocks[-1].proj.in_features
self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
```

This addresses the failure of the original lightweight model to converge on the binary fall/no-fall task (loss stayed ~0.69, accuracy ~52%). The pretrained backbone provides meaningful spatiotemporal features and much larger capacity (~32.8M params).

**Input format:**
- SlowFast expects input shaped `(B, C, T, H, W)`
- Slow pathway samples every 4th frame: `x[:, :, ::4, :, :]`
- Fast pathway uses the original frame rate
- Both are passed as a list `[slow, fast]`

**Dependencies:**
- `pytorchvideo>=0.1.5`
- `Pillow>=9.0`

### 7. Teacher Training Results

Three teacher models were trained on the Kaggle Fall binary classification dataset (Fall / No Fall):

| Teacher | Best F1 | Best Epoch Range | Precision | Recall | Checkpoint |
|---|---|---|---|---|---|
| SlowFast (R50) | 1.0000 | 36-40 | 1.0000 | 1.0000 | `checkpoints/teacher_slowfast_best.pt` |
| VideoSwin (3D-T) | 0.9945 | 26-30 | 0.9906 | 0.9984 | `checkpoints/teacher_video_swin_best.pt` |
| MViT (v2-S) | 1.0000 | 21-25 | 1.0000 | 1.0000 | `checkpoints/teacher_mvit_best.pt` |

Validation set size: 1398 samples (631 fall samples).

Based on these results, 50 epochs is sufficient; all teachers converge well before 50 epochs.

### Usage

```bash
# Step 1: Pre-decode (one-time, ~5-10 min)
python scripts/predecode_videos.py --data_root ./data/kaggle_fall

# Step 2: Install new dependency
pip install pytorchvideo Pillow

# Step 3: Train teachers (50 epochs is sufficient)
python scripts/train_teachers.py --teacher slowfast --batch_size 8 --epochs 50 --lr 1e-4
python scripts/train_teachers.py --teacher video_swin --batch_size 4 --epochs 50 --lr 1e-4
python scripts/train_teachers.py --teacher mvit --batch_size 4 --epochs 50 --lr 1e-4
```
