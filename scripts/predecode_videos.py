"""预解码视频为图片帧

将 videos/ 下的所有视频解码为图片，保存到 frames/ 目录。
训练时直接读图片，避免 cv2.VideoCapture 的同步 IO 阻塞。

用法：
    python scripts/predecode_videos.py --data_root ./data/kaggle_fall --seq_len 16 --input_size 224
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

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # 读取失败，用上一帧或黑图
            if i > 0:
                continue
            frame = np.zeros((input_size, input_size, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (input_size, input_size))
        save_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(save_path, frame)

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="预解码视频为图片帧")
    parser.add_argument("--data_root", type=str, default="./data/kaggle_fall")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4, help="并行解码进程数")
    args = parser.parse_args()

    videos_dir = os.path.join(args.data_root, "videos")
    frames_dir = os.path.join(args.data_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 收集所有视频
    video_files = []
    for f in os.listdir(videos_dir):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(f)

    print(f"[Predecode] Found {len(video_files)} videos in {videos_dir}")
    print(f"[Predecode] Output: {frames_dir}")
    print(f"[Predecode] seq_len={args.seq_len}, input_size={args.input_size}")

    # 逐个解码（简单可靠，避免多进程冲突）
    success = 0
    for vf in tqdm(video_files, desc="Decoding"):
        video_path = os.path.join(videos_dir, vf)
        video_name = os.path.splitext(vf)[0]
        output_dir = os.path.join(frames_dir, video_name)

        # 跳过已解码的
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= args.seq_len:
            success += 1
            continue

        if predecode_video(video_path, output_dir, args.seq_len, args.input_size):
            success += 1

    print(f"[Predecode] Done: {success}/{len(video_files)} videos decoded")
    print(f"[Predecode] Frames saved to: {frames_dir}")


if __name__ == "__main__":
    main()
