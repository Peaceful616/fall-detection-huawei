"""为预解码帧批量提取 COCO 17 关键点

用 YOLOv8s-pose（COCO 17 关键点预训练）对 frames/<stem>/frame_0000..0015.jpg
逐帧推理，输出 CSV 文件 keypoints/<stem>.csv，格式与 Kaggle Fall 对齐：

    Frame,Keypoint,X,Y,Confidence
    1,Nose,x,y,conf
    1,Left Eye,x,y,conf
    ...
    16,Right Ankle,x,y,conf

X/Y 为 224x224 输入图像上的像素坐标（与 _load_keypoints 的归一化逻辑对齐，
_load_keypoints 会按 orig_w/orig_h=224 归一化到 [0,1]）。

Frame 列 1-indexed（CSV 从 1 开始，转 0-indexed 在 _load_keypoints 里做）。

用法：
    python scripts/extract_keypoints.py \\
        --data_root ./data/omnifall_syn \\
        --num_workers 8

断点续传：跳过已存在的 keypoints/<stem>.csv（且非空）。
"""
import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# COCO 17 关键点顺序（与 data/dataset.py::COCO_KP_NAMES 对齐）
COCO_KP_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]


def extract_one_video(task):
    """worker：对单个视频的 16 帧提取关键点，写 CSV

    task: (frames_dir, stem, kp_out_dir, seq_len, input_size, force)
    返回: (stem, status) status in {ok, skip, fail}
    """
    frames_dir, stem, kp_out_dir, seq_len, input_size, force = task
    out_csv = os.path.join(kp_out_dir, f"{stem}.csv")

    # resume：已存在且非空，跳过
    if not force and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return (stem, "skip")

    # 延迟 import + 加载模型，每个 worker 一次（避免主进程加载后 fork 序列化失败）
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[FAIL] {stem}: ultralytics import {e}", flush=True)
        return (stem, "fail")

    # YOLOv8s-pose 权重路径：项目根目录可能没有，ultralytics 会自动下载
    # 第一次运行需要联网，后续走缓存
    try:
        model = YOLO("yolov8s-pose.pt")
    except Exception as e:
        print(f"[FAIL] {stem}: model load {e}", flush=True)
        return (stem, "fail")

    # 读取 16 帧
    frame_paths = [os.path.join(frames_dir, stem, f"frame_{i:04d}.jpg")
                   for i in range(seq_len)]
    missing = [p for p in frame_paths if not os.path.exists(p)]
    if missing:
        print(f"[WARN] {stem}: {len(missing)} frames missing", flush=True)
        return (stem, "fail")

    try:
        # YOLOv8-pose 推理：results 是 list[Results]，每个 Results.keypoints
        # 形状 (n_persons, 17, 3) — x, y, conf
        # 我们取 conf 最大的那个人作为目标（跌倒视频里通常只有一个人）
        results = model(frame_paths, verbose=False)
    except Exception as e:
        print(f"[FAIL] {stem}: inference {e}", flush=True)
        return (stem, "fail")

    # 写 CSV：Frame(1-indexed), Keypoint, X, Y, Confidence
    tmp_csv = out_csv + ".tmp"
    try:
        with open(tmp_csv, "w", encoding="utf-8") as f:
            f.write("Frame,Keypoint,X,Y,Confidence\n")
            for fi, r in enumerate(results):
                frame_idx = fi + 1  # 1-indexed
                kpts = r.keypoints  # Keypoints object
                if kpts is None or len(kpts.data) == 0:
                    # 该帧未检测到人，全零（_load_keypoints 会自动过滤）
                    for kp_name in COCO_KP_NAMES:
                        f.write(f"{frame_idx},{kp_name},0,0,0\n")
                    continue
                # 取 conf 最大的那个人
                # kpts.data shape: (n_persons, 17, 3) — x, y, conf
                kpts_arr = kpts.data.cpu().numpy()  # (n, 17, 3)
                confs = kpts_arr[:, :, 2]  # (n, 17)
                # 用关键点平均 conf 选人
                person_idx = int(confs.mean(axis=1).argmax())
                kp = kpts_arr[person_idx]  # (17, 3)
                for ki, kp_name in enumerate(COCO_KP_NAMES):
                    x, y, conf = float(kp[ki, 0]), float(kp[ki, 1]), float(kp[ki, 2])
                    # 限制到 224 范围内（YOLOv8 可能给越界值，截断到合法范围）
                    x = max(0.0, min(input_size - 1, x))
                    y = max(0.0, min(input_size - 1, y))
                    f.write(f"{frame_idx},{kp_name},{x:.2f},{y:.2f},{conf:.4f}\n")
        # 原子 rename
        if os.path.exists(out_csv):
            os.remove(out_csv)
        os.rename(tmp_csv, out_csv)
        return (stem, "ok")
    except Exception as e:
        print(f"[FAIL] {stem}: write csv {e}", flush=True)
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        return (stem, "fail")


def main():
    parser = argparse.ArgumentParser(description="为预解码帧提取 COCO 17 关键点")
    parser.add_argument("--data_root", type=str, default="./data/omnifall_syn",
                        help="数据集根目录，需含 frames/<stem>/frame_*.jpg")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224,
                        help="预解码帧的分辨率（YOLOv8 输出坐标会归到这个范围）")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="并行进程数（每个进程独立加载 YOLOv8s-pose 模型）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新提取（忽略已有 CSV）")
    args = parser.parse_args()

    frames_dir = os.path.join(args.data_root, "frames")
    kp_out_dir = os.path.join(args.data_root, "keypoints")
    os.makedirs(kp_out_dir, exist_ok=True)

    # 扫描所有视频 stem（从 frames/ 子目录推）
    if not os.path.isdir(frames_dir):
        print(f"[FAIL] frames dir not found: {frames_dir}")
        print(f"       先跑 predecode_videos.py --data_root {args.data_root} --mode omnifall")
        sys.exit(1)

    stems = sorted([d for d in os.listdir(frames_dir)
                    if os.path.isdir(os.path.join(frames_dir, d))])
    print(f"[KP] Found {len(stems)} video frame dirs in {frames_dir}")
    print(f"[KP] Output: {kp_out_dir}")
    print(f"[KP] workers={args.num_workers} seq_len={args.seq_len} input_size={args.input_size}")

    tasks = [(frames_dir, stem, kp_out_dir, args.seq_len, args.input_size, args.force)
             for stem in stems]

    with Pool(processes=args.num_workers) as pool:
        success = skipped = failed = 0
        for stem, status in tqdm(pool.imap_unordered(extract_one_video, tasks),
                                 total=len(tasks), desc="KP"):
            if status == "ok":
                success += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1

    total = len(stems)
    print(f"\n[KP] Done:")
    print(f"  success:  {success}/{total}")
    print(f"  skipped:  {skipped} (already extracted)")
    print(f"  failed:   {failed}")
    print(f"  csv saved to: {kp_out_dir}")
    if failed > 0:
        print(f"[NOTE] {failed} videos failed, rerun to retry them")


if __name__ == "__main__":
    main()
