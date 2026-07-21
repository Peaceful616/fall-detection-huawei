"""bafd6 数据集加载器：单帧图片 + bbox 检测

适配方案 A：把单帧图片"伪视频化"（T=1），输出 bbox 检测
- 输入: (B, 1, 3, H, W) - 单帧伪视频
- 输出: bbox (B, num_boxes, 5) + class (B, num_boxes, num_classes)
"""
import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BAFD6Dataset(Dataset):
    """bafd6 数据集加载器（单帧检测）

    输出:
        image:  (1, 3, H, W) - 伪视频化（T=1）
        boxes:  (N, 5)  [class_id, cx, cy, w, h] 归一化
        n_boxes: int
        label:  int (1=fall, 0=adl)
        image_path: str
    """

    def __init__(self, anno_file: str, images_dir: str, input_size=(288, 288),
                 is_train: bool = True, augment=None):
        self.anno_file = anno_file
        self.images_dir = images_dir
        self.input_size = input_size
        self.is_train = is_train
        self.augment = augment

        with open(anno_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        print(f"[BAFD6 {'train' if is_train else 'val'}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, s["image"])
        img = cv2.imread(img_path)
        if img is None:
            # 容错：返回全黑图
            img = np.zeros((*self.input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        # resize 到 input_size
        img = cv2.resize(img, self.input_size)
        # 归一化 + to tensor (3, H, W)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        # 伪视频化：T=1
        img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

        # boxes
        boxes = []
        for b in s["boxes"]:
            boxes.append([
                b["class_id"],
                b["cx"], b["cy"], b["w"], b["h"]
            ])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        return {
            "video": img_t,  # (1, 3, H, W) 伪视频
            "boxes": boxes,
            "n_boxes": len(boxes),
            "label": 1 if s["label"] == "fall" else 0,
            "image_path": s["image"],
        }


def collate_fn(batch):
    """自定义 batch 整理：boxes 长度不一，pad 对齐"""
    videos = torch.stack([b["video"] for b in batch])  # (B, 1, 3, H, W)
    labels = torch.tensor([b["label"] for b in batch])
    n_boxes = [b["n_boxes"] for b in batch]
    max_n = max(n_boxes) if n_boxes else 1
    # pad boxes
    boxes_padded = []
    for b in batch:
        n = b["n_boxes"]
        if n == 0:
            padded = torch.zeros((max_n, 5))
        else:
            pad = torch.zeros((max_n - n, 5))
            padded = torch.cat([b["boxes"], pad], dim=0)
        boxes_padded.append(padded)
    boxes = torch.stack(boxes_padded)  # (B, max_n, 5)
    return {
        "video": videos,
        "boxes": boxes,
        "n_boxes": torch.tensor(n_boxes),
        "label": labels,
        "image_path": [b["image_path"] for b in batch],
    }


def build_bafd6_datasets(cfg, data_root="./data/bafd6_adapted"):
    """构建 bafd6 训练/验证数据集"""
    train_anno = os.path.join(data_root, "annotations_train.json")
    val_anno = os.path.join(data_root, "annotations_val.json")
    images_dir = os.path.join(data_root, "images")
    train_set = BAFD6Dataset(train_anno, images_dir, cfg.input_size, is_train=True)
    val_set = BAFD6Dataset(val_anno, images_dir, cfg.input_size, is_train=False)
    return train_set, val_set
