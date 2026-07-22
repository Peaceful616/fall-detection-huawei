"""视频数据集：支持 OmniFall / URFD / Multicam 等公开数据集

统一标签格式（5 分类）：
0: ADL          日常活动
1: Fall         跌倒
2: Fall-like    类跌倒（如下蹲）
3: Lying        躺卧
4: Transition   过渡（如起身）
"""
import os
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .augment import LongTailAugment

# COCO 17 关键点顺序（与 Kaggle Fall CSV 的 Keypoint 列名对齐）
COCO_KP_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]

LABEL_MAP = {
    "ADL": 0,
    "Fall": 1,
    "Fall-like": 2,
    "Lying": 3,
    "Transition": 4,
    # 兼容 OmniFall 常见命名
    "fall": 1, "fall_like": 2, "lying": 3, "adl": 0, "transition": 4,
    "sit": 0, "bend": 0, "walk": 0,
}


def load_omnifall(data_root: str) -> List[Dict]:
    """加载 OmniFall 数据集（HuggingFace 上的 simplexsigil2/omnifall）

    假定目录结构：
        data_root/
          ├── annotations.json   # [{video: "xxx.mp4", label: "fall", start: 0.5, end: 3.2}, ...]
          └── videos/
                ├── 001.mp4
                └── ...

    实际下载后如格式不同，可在此处适配。
    """
    anno_file = os.path.join(data_root, "annotations.json")
    videos_dir = os.path.join(data_root, "videos")
    samples = []
    if not os.path.exists(anno_file):
        return samples
    with open(anno_file, "r", encoding="utf-8") as f:
        annos = json.load(f)
    for a in annos:
        vp = os.path.join(videos_dir, a["video"])
        if not os.path.exists(vp):
            continue
        samples.append({
            "video_path": vp,
            "label": LABEL_MAP.get(a.get("label", "adl"), 0),
            "start": a.get("start", 0.0),
            "end": a.get("end", -1.0),
            "scene": a.get("scene", "indoor"),
            "light": a.get("light", "normal"),
        })
    return samples


def load_urfd(data_root: str) -> List[Dict]:
    """加载 UR Fall Detection 数据集"""
    samples = []
    videos_dir = os.path.join(data_root, "videos")
    if not os.path.isdir(videos_dir):
        return samples
    for f in os.listdir(videos_dir):
        if not f.endswith((".mp4", ".avi")):
            continue
        label = 1 if "fall" in f.lower() else 0
        samples.append({
            "video_path": os.path.join(videos_dir, f),
            "label": label,
            "start": 0.0, "end": -1.0,
            "scene": "indoor", "light": "normal",
        })
    return samples


def load_kaggle_fall(data_root: str, split: str = "train") -> List[Dict]:
    """加载 Kaggle Fall Video Dataset（二分类视频 + 可选 keypoint CSV）

    依赖 kaggle_fall_adapter.py 生成的 annotations_{train,val}.json：
        data_root/
        ├── annotations_train.json
        ├── annotations_val.json
        └── videos/
            ├── fall_xxx.mp4
            └── adl_xxx.mp4

    annotation 字段：{video, label, start, end, scene, light, keypoint_csv?}
    返回 samples 字段：{video_path, label(int), start, end, scene, light, keypoint_csv?}
    """
    anno_file = os.path.join(data_root, f"annotations_{split}.json")
    videos_dir = os.path.join(data_root, "videos")
    samples = []
    if not os.path.exists(anno_file):
        return samples
    with open(anno_file, "r", encoding="utf-8") as f:
        annos = json.load(f)
    for a in annos:
        vp = os.path.join(videos_dir, a["video"])
        if not os.path.exists(vp):
            continue
        entry = {
            "video_path": vp,
            "label": LABEL_MAP.get(a.get("label", "adl"), 0),
            "start": a.get("start", 0.0),
            "end": a.get("end", -1.0),
            "scene": a.get("scene", "indoor"),
            "light": a.get("light", "normal"),
        }
        if "keypoint_csv" in a and a["keypoint_csv"]:
            entry["keypoint_csv"] = a["keypoint_csv"]
        samples.append(entry)
    return samples


class VideoFallDataset(Dataset):
    """视频跌倒检测数据集

    每次采样：从视频中随机截取 T 帧，标签为该片段的主导动作
    若 sample 含 keypoint_csv，则同时读取对齐帧的 17 COCO 关键点，
    输出 aux_kp: (T, 17, 2) 归一化坐标（供姿态辅助分支监督）
    """

    def __init__(self, cfg, samples: List[Dict], is_train: bool = True):
        self.cfg = cfg
        self.samples = samples
        self.is_train = is_train
        self.augment = LongTailAugment(cfg) if is_train else None
        self.seq_len = cfg.seq_len
        self.input_size = cfg.input_size
        self.aux_kp_enabled = getattr(cfg, "aux_kp_enabled", False)

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path: str, start: float, end: float):
        """读取视频片段

        返回 (frames (T,H,W,3) RGB in [0,255], start_frame, (orig_w, orig_h))
        start_frame 用于对齐 keypoint CSV
        orig_w/orig_h 为原视频分辨率，用于 keypoint 归一化
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.input_size[0]
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.input_size[1]
        # 截取范围
        start_frame = int(start * fps)
        end_frame = int(end * fps) if end > 0 else total
        # 随机选起点（保证取到 seq_len 帧）
        if end_frame - start_frame < self.seq_len:
            start_frame = max(0, end_frame - self.seq_len)
        if self.is_train and end_frame - start_frame > self.seq_len:
            offset = np.random.randint(0, end_frame - start_frame - self.seq_len + 1)
            start_frame = start_frame + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # 不足时补最后一帧
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*self.input_size, 3), dtype=np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.input_size)
            frames.append(frame)
        cap.release()
        return np.stack(frames), start_frame, (orig_w, orig_h)

    def _load_keypoints(self, csv_path: str, start_frame: int,
                        orig_w: int, orig_h: int) -> np.ndarray:
        """读取 keypoint CSV，返回 (T, 17, 2) 归一化坐标 [0,1]

        CSV 格式：Frame,Keypoint,X,Y,Confidence
        X/Y 为原视频像素坐标，按 orig_w/orig_h 归一化到 [0,1]
        Frame 为 1-indexed 视频帧号
        """
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # 按 Frame 分组：(0-indexed frame -> {kp_name: (x, y)})
                frame_kp = {}
                for row in reader:
                    try:
                        fi = int(row["Frame"]) - 1  # CSV 从 1 开始，转 0-indexed
                    except (ValueError, KeyError):
                        continue
                    kp_name = row.get("Keypoint", "")
                    if kp_name not in COCO_KP_NAMES:
                        continue
                    try:
                        x = float(row["X"])
                        y = float(row["Y"])
                    except (ValueError, KeyError):
                        continue
                    frame_kp.setdefault(fi, {})[kp_name] = (x, y)

            T = self.seq_len
            out = np.zeros((T, 17, 2), dtype=np.float32)
            for t in range(T):
                fi = start_frame + t
                kps = frame_kp.get(fi, {})
                for k, name in enumerate(COCO_KP_NAMES):
                    if name in kps:
                        x, y = kps[name]
                        out[t, k, 0] = x / max(orig_w, 1)
                        out[t, k, 1] = y / max(orig_h, 1)
            return out
        except Exception:
            # keypoint 读取失败不影响主任务，返回全零
            return np.zeros((self.seq_len, 17, 2), dtype=np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames, start_frame, (orig_w, orig_h) = self._load_video_frames(
            s["video_path"], s["start"], s["end"])
        # to tensor (T, 3, H, W) in [0, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # 长尾增强
        if self.augment is not None:
            x = self.augment(x.unsqueeze(0)).squeeze(0)
        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = {
            "video": x,           # (T, 3, H, W)
            "label": s["label"],
            "video_path": s["video_path"],
            "scene": s.get("scene", "indoor"),
            "light": s.get("light", "normal"),
        }

        # 姿态辅助分支监督信号（仅训练期）
        if self.is_train and self.aux_kp_enabled and "keypoint_csv" in s:
            kp = self._load_keypoints(s["keypoint_csv"], start_frame, orig_w, orig_h)
            out["aux_kp"] = torch.from_numpy(kp)  # (T, 17, 2)
        return out


class VideoFallDatasetFast(Dataset):
    """快速版：读取预解码的帧图片，避免 cv2.VideoCapture 的同步 IO 阻塞

    目录结构（由 scripts/predecode_videos.py 生成）：
        data_root/
        ├── frames/
        │   ├── video_001/
        │   │   ├── frame_0000.jpg
        │   │   ├── frame_0001.jpg
        │   │   └── ...
        │   └── video_002/
        │       └── ...
        └── annotations_train.json

    比 VideoFallDataset 快 5-10x，GPU 利用率显著提升。
    """

    def __init__(self, cfg, samples: List[Dict], is_train: bool = True):
        self.cfg = cfg
        self.samples = samples
        self.is_train = is_train
        self.augment = LongTailAugment(cfg) if is_train else None
        self.seq_len = cfg.seq_len
        self.input_size = cfg.input_size
        self.aux_kp_enabled = getattr(cfg, "aux_kp_enabled", False)
        self.frames_dir = os.path.join(cfg.data_root, "frames")

        # 预构建索引：video_name -> frame 目录路径
        self.frame_dirs = {}
        for s in samples:
            video_name = Path(s["video_path"]).stem
            frame_dir = os.path.join(self.frames_dir, video_name)
            self.frame_dirs[s["video_path"]] = frame_dir

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, frame_dir: str) -> np.ndarray:
        """读取预解码的帧图片，返回 (T, H, W, 3) RGB"""
        frames = []
        for i in range(self.seq_len):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                # PIL 读图比 cv2 快，且已经是 RGB
                img = Image.open(frame_path).convert("RGB")
                frames.append(np.array(img))
            else:
                # 不足时补最后一帧或黑图
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*self.input_size, 3), dtype=np.uint8))
        return np.stack(frames)

    def _load_keypoints(self, csv_path: str, start_frame: int,
                        orig_w: int, orig_h: int) -> np.ndarray:
        """读取 keypoint CSV，返回 (T, 17, 2) 归一化坐标 [0,1]"""
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                frame_kp = {}
                for row in reader:
                    try:
                        fi = int(row["Frame"]) - 1
                    except (ValueError, KeyError):
                        continue
                    kp_name = row.get("Keypoint", "")
                    if kp_name not in COCO_KP_NAMES:
                        continue
                    try:
                        x = float(row["X"])
                        y = float(row["Y"])
                    except (ValueError, KeyError):
                        continue
                    frame_kp.setdefault(fi, {})[kp_name] = (x, y)

            T = self.seq_len
            out = np.zeros((T, 17, 2), dtype=np.float32)
            for t in range(T):
                fi = start_frame + t
                kps = frame_kp.get(fi, {})
                for k, name in enumerate(COCO_KP_NAMES):
                    if name in kps:
                        x, y = kps[name]
                        out[t, k, 0] = x / max(orig_w, 1)
                        out[t, k, 1] = y / max(orig_h, 1)
            return out
        except Exception:
            return np.zeros((self.seq_len, 17, 2), dtype=np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame_dir = self.frame_dirs[s["video_path"]]

        # 读取预解码帧（快速）
        frames = self._load_frames(frame_dir)

        # to tensor (T, 3, H, W) in [0, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # 长尾增强
        if self.augment is not None:
            x = self.augment(x.unsqueeze(0)).squeeze(0)

        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = {
            "video": x,
            "label": s["label"],
            "video_path": s["video_path"],
            "scene": s.get("scene", "indoor"),
            "light": s.get("light", "normal"),
        }

        # 姿态辅助分支监督信号
        if self.is_train and self.aux_kp_enabled and "keypoint_csv" in s:
            # 预解码模式下 start_frame 固定为 0（均匀采样）
            orig_w, orig_h = self.input_size[0], self.input_size[1]
            kp = self._load_keypoints(s["keypoint_csv"], 0, orig_w, orig_h)
            out["aux_kp"] = torch.from_numpy(kp)

        return out


def build_datasets(cfg):
    """构建训练/验证数据集

    优先检测预解码帧目录（frames/），存在则用快速版 VideoFallDatasetFast；
    否则回退到原 VideoFallDataset（cv2.VideoCapture 逐帧读取）。
    """
    frames_dir = os.path.join(cfg.data_root, "frames")
    use_fast = os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0

    kaggle_train = os.path.join(cfg.data_root, "annotations_train.json")
    if os.path.exists(kaggle_train):
        train_samples = load_kaggle_fall(cfg.data_root, split="train")
        val_samples = load_kaggle_fall(cfg.data_root, split="val")
        if train_samples and val_samples:
            if use_fast:
                DatasetClass = VideoFallDatasetFast
                print(f"[Dataset] Using fast mode (pre-decoded frames from {frames_dir})")
            else:
                DatasetClass = VideoFallDataset
                print(f"[Dataset] Using slow mode (cv2.VideoCapture). "
                      f"Run: python scripts/predecode_videos.py --data_root {cfg.data_root}")
            train_set = DatasetClass(cfg, train_samples, is_train=True)
            val_set = DatasetClass(cfg, val_samples, is_train=False)
            print(f"[Dataset] Kaggle Fall: train={len(train_set)}, val={len(val_set)}")
            return train_set, val_set

    # 回退：OmniFall + URFD
    all_samples = []
    omnifall_samples = load_omnifall(cfg.data_root)
    all_samples.extend(omnifall_samples)
    urfd_root = os.path.join(os.path.dirname(cfg.data_root), "urfd")
    if os.path.isdir(urfd_root):
        all_samples.extend(load_urfd(urfd_root))
    np.random.seed(42)
    np.random.shuffle(all_samples)
    n_total = len(all_samples)
    n_train = int(n_total * 0.8)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    if use_fast:
        DatasetClass = VideoFallDatasetFast
        print(f"[Dataset] Using fast mode (pre-decoded frames)")
    else:
        DatasetClass = VideoFallDataset
        print(f"[Dataset] Using slow mode (cv2.VideoCapture)")
    train_set = DatasetClass(cfg, train_samples, is_train=True)
    val_set = DatasetClass(cfg, val_samples, is_train=False)
    print(f"[Dataset] OmniFall/URFD fallback: train={len(train_set)}, val={len(val_set)}")
    return train_set, val_set
