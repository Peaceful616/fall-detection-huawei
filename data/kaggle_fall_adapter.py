"""Kaggle Fall Video Dataset 适配器

数据集来源：archive.zip（Kaggle Fall Video Dataset，约 16GB）
结构：
    archive.zip
    ├── Fall/
    │   ├── Raw_Video/20240912_101331.mp4
    │   └── Keypoints_CSV/20240912_101331_keypoints.csv
    └── No_Fall/
        ├── Raw_Video/B_D_0001.mp4
        └── Keypoints_CSV/B_D_0001_keypoints.csv

每个 mp4 配一个 keypoints.csv（17 COCO 关键点 × 每帧）：
    Frame,Keypoint,X,Y,Confidence
    1,Nose,425.47,163.05,0.999
    1,Left Eye,423.16,148.12,0.995
    ...

适配输出（统一到 dataset.py 的接口）：
    out_dir/
    ├── annotations_train.json
    ├── annotations_val.json
    └── videos/
        ├── fall_20240912_101331.mp4
        ├── adl_B_D_0001.mp4
        └── ...

annotations 字段：
    {
        "video": "fall_20240912_101331.mp4",
        "label": "fall",                 # fall / adl
        "start": 0.0, "end": -1.0,
        "scene": "indoor", "light": "normal",
        "keypoint_csv": "fall_20240912_101331_keypoints.csv"  # 可选
    }

label 映射（到 5 分类）：
    Fall    → 1 (Fall)
    No_Fall → 0 (ADL)

注：本数据集为二分类，无 Fall-like/Lying/Transition 细分。
"""
import os
import json
import random
import shutil
from pathlib import Path


# COCO 17 关键点顺序（与 csv 中 Keypoint 列名对齐）
COCO_KP_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]


def adapt_kaggle_fall(src_dir: str, out_dir: str, train_ratio: float = 0.8,
                      copy_videos: bool = False, use_keypoints: bool = True):
    """适配 Kaggle Fall Video Dataset

    src_dir: 解压后的 archive 根目录（含 Fall/ 和 No_Fall/）
    out_dir: 输出目录
    train_ratio: 训练集比例
    copy_videos: True 复制视频到 out_dir/videos；False 建立符号链接（省磁盘）
                 Windows 软链需管理员权限，失败时自动回退复制
    use_keypoints: 是否索引 keypoint CSV（用于姿态辅助分支监督）
    """
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos_out = out / "videos"
    videos_out.mkdir(exist_ok=True)

    # 子目录命名
    fall_video_dir = src / "Fall" / "Raw_Video"
    fall_kp_dir = src / "Fall" / "Keypoints_CSV"
    nofall_video_dir = src / "No_Fall" / "Raw_Video"
    nofall_kp_dir = src / "No_Fall" / "Keypoints_CSV"

    if not fall_video_dir.is_dir() or not nofall_video_dir.is_dir():
        print(f"[FAIL] Expected Fall/Raw_Video/ and No_Fall/Raw_Video/ under {src_dir}")
        return False

    def _link_or_copy(src_file: Path, dst_file: Path):
        """优先软链，失败则复制"""
        if dst_file.exists():
            return
        if copy_videos:
            shutil.copy(src_file, dst_file)
            return
        # Windows 软链需权限，失败回退复制
        try:
            os.symlink(src_file, dst_file)
        except (OSError, NotImplementedError):
            shutil.copy(src_file, dst_file)

    def _scan(video_dir: Path, kp_dir: Path, label: str, prefix: str):
        """扫描一个类别的视频 + keypoint CSV"""
        items = []
        for mp4 in sorted(video_dir.glob("*.mp4")):
            stem = mp4.stem
            dst_name = f"{prefix}_{stem}.mp4"
            dst_path = videos_out / dst_name
            _link_or_copy(mp4, dst_path)

            entry = {
                "video": dst_name,
                "label": label,
                "start": 0.0,
                "end": -1.0,
                "scene": "indoor",
                "light": "normal",
            }
            # keypoint CSV（可选）
            if use_keypoints:
                kp_src = kp_dir / f"{stem}_keypoints.csv"
                if kp_src.exists():
                    # 存绝对路径（正斜杠），dataset.py 读取时用
                    entry["keypoint_csv"] = str(kp_src.resolve()).replace("\\", "/")
            items.append(entry)
        return items

    print(f"[Scan] Fall videos from {fall_video_dir}")
    fall_items = _scan(fall_video_dir, fall_kp_dir, "fall", "fall")
    print(f"  -> {len(fall_items)} fall samples")

    print(f"[Scan] No_Fall videos from {nofall_video_dir}")
    nofall_items = _scan(nofall_video_dir, nofall_kp_dir, "adl", "adl")
    print(f"  -> {len(nofall_items)} adl samples")

    samples = fall_items + nofall_items
    print(f"\n[Total] {len(samples)} samples (fall={len(fall_items)}, adl={len(nofall_items)})")

    # 划分 train/val（seed=42，与 bafd6_adapter 一致，可复现）
    random.seed(42)
    random.shuffle(samples)
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    # 写 annotations
    out_train = out / "annotations_train.json"
    out_val = out / "annotations_val.json"
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    n_kp_train = sum(1 for s in train_samples if "keypoint_csv" in s)
    n_kp_val = sum(1 for s in val_samples if "keypoint_csv" in s)

    print(f"\n[Adapt Done]")
    print(f"  Train: {len(train_samples)} samples ({n_kp_train} with keypoints) -> {out_train}")
    print(f"  Val:   {len(val_samples)} samples ({n_kp_val} with keypoints) -> {out_val}")
    print(f"  Videos dir: {videos_out}")
    return True


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "./data/kaggle_fall_raw"
    out = sys.argv[2] if len(sys.argv) > 2 else "./data/kaggle_fall"
    adapt_kaggle_fall(src, out)
