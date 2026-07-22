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

### Usage

```bash
# Step 1: Pre-decode (one-time, ~5-10 min)
python scripts/predecode_videos.py --data_root ./data/kaggle_fall

# Step 2: Train (auto-detects pre-decoded frames)
python scripts/train_teachers.py --teacher slowfast --batch_size 8 --epochs 100
```
