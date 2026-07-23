"""预解码视频为图片帧

将视频解码为 seq_len 帧图片，保存到 frames/ 目录。
训练时直接读图片，避免 cv2.VideoCapture 的同步 IO 阻塞。

支持两种模式：
1. kaggle 模式（默认）：扫描 data_root/videos/ 下所有视频
   python scripts/predecode_videos.py --data_root ./data/kaggle_fall
2. omnifall 模式：从 data_root/annotations/annotations_*.json 读 path
   列表，视频在 data_root/videos/ 下，文件名 = path 最后一段 + .mp4
   python scripts/predecode_videos.py --data_root ./data/omnifall_syn --mode omnifall

断点续传：跳过已完整解码的目录（帧数 >= seq_len），中断后重跑即可。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def predecode_video(video_path: str, output_dir: str, seq_len: int, input_size: int):
    """将视频解码为帧图片，每个视频保存 seq_len 帧（均匀采样）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    # 均匀采样 seq_len 帧
    if total_frames >= seq_len:
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    else:
        # 不足时重复最后一帧
        indices = list(range(total_frames)) + [total_frames - 1] * (seq_len - total_frames)

    os.makedirs(output_dir, exist_ok=True)

    # 先写临时目录，完成后原子 rename，避免半成品
    tmp_dir = output_dir + ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    ok = True
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # 读取失败，用上一帧或黑图
            if i > 0:
                # 复用上一帧
                prev_path = os.path.join(tmp_dir, f"frame_{i-1:04d}.jpg")
                if os.path.exists(prev_path):
                    frame = cv2.imread(prev_path)
                else:
                    frame = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            else:
                frame = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (input_size, input_size))
        save_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(save_path, frame)

    cap.release()

    # 完成后原子 rename（删旧目录再移）
    if os.path.exists(output_dir):
        # 已存在的可能是上次半成品，删掉
        import shutil
        shutil.rmtree(output_dir)
    os.rename(tmp_dir, output_dir)
    return True


def collect_kaggle_videos(videos_dir: str):
    """kaggle 模式：扫描 videos/ 下所有视频文件，返回 [(video_path, output_stem), ...]"""
    out = []
    if not os.path.isdir(videos_dir):
        return out
    for f in os.listdir(videos_dir):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(videos_dir, f)
            stem = os.path.splitext(f)[0]
            out.append((video_path, stem))
    return out


def collect_omnifall_videos(data_root: str, videos_dir: str, splits=None):
    """omnifall 模式：从 annotations 读 path 列表

    path 是逻辑 ID（如 'adl/HopS6'），视频文件在 videos/ 下，
    文件名 = path 最后一段 + '.mp4'（如 'HopS6.mp4'）

    返回 [(video_path, output_stem), ...]
    output_stem 用 path 最后一段（去斜杠），保持与 kaggle 一致
    """
    anno_dir = os.path.join(data_root, "annotations")
    if not os.path.isdir(anno_dir):
        print(f"[FAIL] annotations dir not found: {anno_dir}")
        print(f"       先跑 data/omnifall_adapter.py --config of-syn --out {data_root}")
        return []

    if splits is None:
        splits = ["train", "validation", "test"]

    seen = set()  # path 去重（同一视频可能多段标注）
    out = []
    missing = 0
    for split in splits:
        anno_file = os.path.join(anno_dir, f"annotations_{split}.json")
        if not os.path.exists(anno_file):
            continue
        with open(anno_file, "r", encoding="utf-8") as f:
            annos = json.load(f)
        for a in annos:
            logical_path = a["video_path"]  # 如 'adl/HopS6'
            if logical_path in seen:
                continue
            seen.add(logical_path)
            # 文件名 = path 最后一段 + .mp4
            stem = logical_path.split("/")[-1]
            video_path = os.path.join(videos_dir, f"{stem}.mp4")
            if not os.path.exists(video_path):
                missing += 1
                # 跳过缺失的，但仍记录（后面统计）
                continue
            out.append((video_path, stem))
    if missing:
        print(f"[WARN] {missing} videos referenced in annotations not found in {videos_dir}")
    return out


def main():
    parser = argparse.ArgumentParser(description="预解码视频为图片帧")
    parser.add_argument("--data_root", type=str, default="./data/kaggle_fall")
    parser.add_argument("--mode", choices=["kaggle", "omnifall"], default="kaggle",
                        help="kaggle=扫 videos/ 目录；omnifall=读 annotations/ 的 path")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4, help="并行解码进程数（暂未用，单进程）")
    args = parser.parse_args()

    videos_dir = os.path.join(args.data_root, "videos")
    frames_dir = os.path.join(args.data_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 收集视频列表
    if args.mode == "omnifall":
        video_list = collect_omnifall_videos(args.data_root, videos_dir)
    else:
        video_list = collect_kaggle_videos(videos_dir)

    print(f"[Predecode] mode={args.mode}")
    print(f"[Predecode] Found {len(video_list)} videos in {videos_dir}")
    print(f"[Predecode] Output: {frames_dir}")
    print(f"[Predecode] seq_len={args.seq_len}, input_size={args.input_size}")

    # 逐个解码（断点续传：跳过已完整解码的）
    success = 0
    skipped = 0
    failed = 0
    for video_path, stem in tqdm(video_list, desc="Decoding"):
        output_dir = os.path.join(frames_dir, stem)
        # 跳过已完整解码的（帧数 >= seq_len）
        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) >= args.seq_len:
            skipped += 1
            continue
        # 清理半成品临时目录
        tmp_dir = output_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        try:
            if predecode_video(video_path, output_dir, args.seq_len, args.input_size):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[WARN] {stem}: {e}")
            failed += 1
            # 清理半成品
            if os.path.isdir(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)

    total = len(video_list)
    print(f"\n[Predecode] Done:")
    print(f"  success:  {success}/{total}")
    print(f"  skipped:  {skipped} (already decoded)")
    print(f"  failed:   {failed}")
    print(f"  frames saved to: {frames_dir}")
    if success + skipped < total:
        print(f"[NOTE] {failed} videos failed, rerun to retry them")


if __name__ == "__main__":
    main()
