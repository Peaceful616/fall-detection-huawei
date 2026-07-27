"""阈值扫描：在 student_best.pt 上扫 binary decision threshold

学生模型输出 5 类 logits，正常 argmax 预测后合并 fall+fall-like → positive。
但 argmax 是硬决策，没利用概率信息。

本脚本扫阈值：对每个样本计算 P(fall) + P(fall-like)，超过阈值则判 fall。
扫 0.1-0.9 步长 0.05，输出每个阈值的 P/R/F1，找 binary F1 最高的点。

用法：
    python scripts/scan_threshold.py \\
        --ckpt /root/autodl-tmp/fall-detection-huawei/checkpoints/student_best.pt \\
        --data_root ./data/omnifall_syn
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import cfg
from models.student import build_student
from data.dataset import build_datasets
from utils.metrics import compute_fall_detection_metrics


@torch.no_grad()
def collect_logits(student, loader, device):
    """跑 val 集，收集每个样本的 5 类 softmax 概率和 GT label"""
    student.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Val inference"):
        x = batch["video"].to(device)
        y = batch["label"]
        out = student(x)
        logits = out["logits"]  # (B, 5)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def scan_thresholds(probs, labels):
    """扫阈值 0.1-0.9 步长 0.05

    binary target: label in {1, 2} (Fall, Fall-like) → positive
    binary score: P(1) + P(2)
    pred = score >= threshold
    """
    binary_targets = np.isin(labels, [1, 2]).astype(int)
    fall_like_score = probs[:, 1] + probs[:, 2]  # P(Fall) + P(Fall-like)
    results = []
    print(f"\n{'Thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 50)
    best_f1 = 0
    best_thresh = 0
    for t in np.arange(0.05, 1.0, 0.05):
        pred = (fall_like_score >= t).astype(int)
        tp = int(((pred == 1) & (binary_targets == 1)).sum())
        fp = int(((pred == 1) & (binary_targets == 0)).sum())
        fn = int(((pred == 0) & (binary_targets == 1)).sum())
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        results.append({"thresh": float(t), "precision": p, "recall": r,
                        "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        print(f"{t:>7.2f} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {tp:>5d} {fp:>5d} {fn:>5d}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    print("-" * 50)
    print(f"[BEST] thresh={best_thresh:.2f} F1={best_f1:.4f}")
    return results, best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser(description="扫描 binary decision threshold")
    parser.add_argument("--ckpt", type=str,
                        default="/root/autodl-tmp/fall-detection-huawei/checkpoints/student_best.pt",
                        help="student_best.pt 路径")
    parser.add_argument("--data_root", type=str, default="./data/omnifall_syn")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 覆盖 cfg.data_root（远程可能没拉到最新 default.py）
    cfg.data_root = args.data_root

    # 数据
    _, val_set = build_datasets(cfg)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
    print(f"[Val] {len(val_set)} samples")

    # 模型
    student = build_student(cfg).to(device)
    if not os.path.exists(args.ckpt):
        print(f"[FAIL] ckpt not found: {args.ckpt}")
        sys.exit(1)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["state_dict"])
    print(f"[Loaded] {args.ckpt} (epoch={ckpt.get('epoch')}, f1={ckpt.get('f1'):.4f})")

    # 推理
    probs, labels = collect_logits(student, val_loader, device)
    print(f"[Logits] shape={probs.shape}")

    # 扫阈值
    results, best_thresh, best_f1 = scan_thresholds(probs, labels)

    # 对比 argmax 基线
    print("\n[Compare] argmax baseline (current deployment):")
    pred_cls = probs.argmax(axis=1)
    baseline = compute_fall_detection_metrics(pred_cls.tolist(), labels.tolist())
    print(f"  argmax  P={baseline['precision']:.4f} R={baseline['recall']:.4f} F1={baseline['f1']:.4f}")
    print(f"  thresh  P={best_f1:.4f} (improvement: {best_f1 - baseline['f1']:+.4f})")


if __name__ == "__main__":
    main()
