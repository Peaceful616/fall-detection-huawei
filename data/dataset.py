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

from .augment import LongTailAugment

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


class VideoFallDataset(Dataset):
    """视频跌倒检测数据集

    每次采样：从视频中随机截取 T 帧，标签为该片段的主导动作
    """

    def __init__(self, cfg, samples: List[Dict], is_train: bool = True):
        self.cfg = cfg
        self.samples = samples
        self.is_train = is_train
        self.augment = LongTailAugment(cfg) if is_train else None
        self.seq_len = cfg.seq_len
        self.input_size = cfg.input_size

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path: str, start: float, end: float) -> np.ndarray:
        """读取视频片段，返回 (T, H, W, 3) RGB in [0, 255]"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        return np.stack(frames)  # (T, H, W, 3)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = self._load_video_frames(s["video_path"], s["start"], s["end"])
        # to tensor (T, 3, H, W) in [0, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # 长尾增强
        if self.augment is not None:
            x = self.augment(x.unsqueeze(0)).squeeze(0)
        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        return {
            "video": x,           # (T, 3, H, W)
            "label": s["label"],
            "video_path": s["video_path"],
            "scene": s.get("scene", "indoor"),
            "light": s.get("light", "normal"),
        }


def build_datasets(cfg):
    """构建训练/验证数据集"""
    all_samples = []
    # OmniFall
    omnifall_samples = load_omnifall(cfg.data_root)
    all_samples.extend(omnifall_samples)
    # URFD（如有）
    urfd_root = os.path.join(os.path.dirname(cfg.data_root), "urfd")
    if os.path.isdir(urfd_root):
        all_samples.extend(load_urfd(urfd_root))
    # 划分
    np.random.seed(42)
    np.random.shuffle(all_samples)
    n_total = len(all_samples)
    n_train = int(n_total * 0.8)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    train_set = VideoFallDataset(cfg, train_samples, is_train=True)
    val_set = VideoFallDataset(cfg, val_samples, is_train=False)
    print(f"[Dataset] train={len(train_set)}, val={len(val_set)}")
    return train_set, val_set
