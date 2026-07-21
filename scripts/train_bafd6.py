"""bafd6 数据集训练脚本（方案 A：单帧检测）

简化版训练：
- 任务：单帧图片 → bbox 检测
- backbone: YOLOv8s 全解冻
- head: 检测头（bbox + cls + obj）
- 损失: YOLO 风格 CIoU + BCE + CE
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.default import cfg
from models.detector import build_detector
from data.bafd6_dataset import build_bafd6_datasets, collate_fn
from utils.det_loss import DetectionLoss
from utils.metrics import compute_fall_detection_metrics, print_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total = {"total": 0, "box": 0, "obj": 0, "cls": 0}
    n = 0
    for i, batch in enumerate(loader):
        x = batch["video"].to(device)  # (B, 1, 3, H, W)
        targets = {
            "boxes": batch["boxes"].to(device),
            "n_boxes": batch["n_boxes"].to(device),
        }
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, targets)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for k in total:
            total[k] += loss[k] * x.size(0)
        n += x.size(0)
        if i % 10 == 0:
            print(f"  [Epoch {epoch+1} Batch {i}] "
                  f"total={loss['total'].item():.4f} "
                  f"box={loss['box']:.4f} obj={loss['obj']:.4f} "
                  f"cls={loss['cls']:.4f}", flush=True)
    msg = "  ".join(f"{k}={v/n:.4f}" for k, v in total.items())
    print(f"  [Epoch {epoch+1} Avg] {msg}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        x = batch["video"].to(device)
        out = model(x)
        # 用 obj > 0.5 判定是否检测到
        for b in range(x.size(0)):
            obj_pred = torch.sigmoid(out["obj"][b])  # (A,)
            detected = (obj_pred > 0.5).any().item()
            n_gt = int(batch["n_boxes"][b].item())
            preds.append(1 if detected else 0)
            targets.append(1 if n_gt > 0 else 0)
    metrics = compute_fall_detection_metrics(preds, targets)
    print_metrics(metrics, "bafd6 Validation")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", default="./data/bafd6_adapted")
    parser.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Device] {device}")

    # 数据
    train_set, val_set = build_bafd6_datasets(cfg, args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_fn)

    # 模型
    model = build_detector(cfg).to(device)
    stats = model.count_parameters()
    print(f"[Detector] params={stats['total_M']}M "
          f"fp32={stats['fp32_MB']}MB int8={stats['int8_MB']}MB")
    assert stats['total'] <= 20e6, "Params exceed 20M"

    # 优化器（backbone 用更小学习率）
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = DetectionLoss(num_anchors=3, num_classes=1)

    # 训练
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                ckpt = os.path.join(args.save_dir, "detector_bafd6_best.pt")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "f1": best_f1,
                    "config": cfg.__dict__,
                }, ckpt)
                print(f"  Saved: {ckpt} (F1={best_f1:.4f})")

    print(f"\n[Done] Best F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
