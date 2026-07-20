"""推理 demo：视频 → 跌倒检测 + 报警

使用流程：
    python scripts/infer_demo.py --video test.mp4 --ckpt student_best.pt

无权重时也能跑（随机初始化），用于演示推理链路
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch

from configs.default import cfg
from models.student import build_student
from utils.postprocess import FallAlarmPostprocess, Box

LABEL_NAMES = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]


def load_model(ckpt_path: str, device):
    student = build_student(cfg).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[OK] Loaded: {ckpt_path}")
    else:
        print("[WARN] No ckpt, use random init (demo only)")
    student.eval()
    return student


def preprocess_frame(frame, input_size):
    """BGR → tensor (3, H, W) normalized"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, input_size)
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def draw_overlay(frame, p_fall, alarm, label):
    """在帧上叠加信息"""
    h, w = frame.shape[:2]
    # 跌倒概率条
    bar_w = int(w * 0.4)
    bar_h = 20
    cv2.rectangle(frame, (10, 10), (10 + bar_w, 10 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 10), (10 + int(bar_w * p_fall), 10 + bar_h),
                 (0, 0, 255) if p_fall > 0.5 else (0, 255, 0), -1)
    cv2.putText(frame, f"Fall Prob: {p_fall:.2f}", (10 + bar_w + 10, 26),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # 标签
    cv2.putText(frame, f"Label: {label}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # 报警
    if alarm:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "!! FALL DETECTED !!", (w // 2 - 200, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "student_best.pt"))
    parser.add_argument("--output", default="demo_output.mp4")
    parser.add_argument("--show", action="store_true", help="实时显示（无 GUI 环境忽略）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    postprocess = FallAlarmPostprocess()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FAIL] Cannot open video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {W}x{H} @ {fps:.1f}fps, total={total}")

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    # 滑窗
    frame_buffer = []
    frame_idx = 0

    print("[Run] Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        t = preprocess_frame(frame, cfg.input_size)
        frame_buffer.append(t)
        if len(frame_buffer) > cfg.seq_len:
            frame_buffer.pop(0)

        if len(frame_buffer) < cfg.seq_len:
            out_writer.write(frame)
            frame_idx += 1
            continue

        # 滑窗推理
        x = torch.stack(frame_buffer).unsqueeze(0).to(device)  # (1, T, 3, H, W)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out["logits"], dim=1)[0]
            p_fall = probs[1].item() + probs[2].item()  # Fall + Fall-like
            label = LABEL_NAMES[probs.argmax().item()]

        # 简化：用整图 bounding box 模拟（实际接 YOLO 人体检测）
        box = Box(0, 0, W, H)
        alarm = postprocess.update(p_fall, [box], frame_idx)

        # 叠加信息
        frame = draw_overlay(frame, p_fall, alarm, label)
        out_writer.write(frame)

        if args.show:
            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total}, p_fall={p_fall:.2f}, label={label}")

    cap.release()
    out_writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[Done] Output: {args.output}")


if __name__ == "__main__":
    main()
