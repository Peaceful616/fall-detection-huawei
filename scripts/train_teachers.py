"""教师模型训练脚本

三个教师独立训练，可单独执行
"""
import argparse
import os
import sys

# 添加项目根目录到 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.default import cfg
from models.teachers import build_teacher
from data.dataset import build_datasets
from utils.metrics import accuracy, compute_fall_detection_metrics, print_metrics


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_name=""):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for batch in loader:
        x = batch["video"].to(device)  # (B, T, 3, H, W)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(x)
        if "modal_logits" in out:
            # MViT: 主任务 + 模态对抗
            loss = criterion(out["logits"], y)
        else:
            loss = criterion(out["logits"], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(out["logits"], y) * x.size(0)
        n += x.size(0)
    print(f"  [{teacher_name}] loss={total_loss/n:.4f} acc={total_acc/n:.4f}")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, scenes = [], [], []
    for batch in loader:
        x = batch["video"].to(device)
        y = batch["label"]
        out = model(x)
        pred = out["logits"].argmax(dim=1).cpu()
        preds.extend(pred.tolist())
        targets.extend(y.tolist())
        scenes.extend(batch["scene"])
    metrics = compute_fall_detection_metrics(preds, targets)
    scene_metrics = {}  # 简化
    print_metrics(metrics, "Validation")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["slowfast", "video_swin", "mvit"], required=True)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)

    # 模型
    model = build_teacher(args.teacher, num_classes=cfg.num_classes).to(device)
    print(f"[Teacher: {args.teacher}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device, args.teacher)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, val_loader, device)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                ckpt_path = os.path.join(args.save_dir, f"teacher_{args.teacher}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "f1": best_f1,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
