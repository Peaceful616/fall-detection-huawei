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
from torch.cuda.amp import autocast, GradScaler

from configs.default import cfg
from models.teachers import build_teacher
from data.dataset import build_datasets
from utils.metrics import accuracy, compute_fall_detection_metrics, \
    compute_multiclass_metrics, print_metrics


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_name="", scaler=None):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for batch in loader:
        x = batch["video"].to(device, non_blocking=True)  # (B, T, 3, H, W)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()
        # AMP 混合精度
        with autocast():
            out = model(x)
            loss = criterion(out["logits"], y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(out["logits"], y) * x.size(0)
        n += x.size(0)
    print(f"  [{teacher_name}] loss={total_loss/n:.4f} acc={total_acc/n:.4f}")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 5):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        x = batch["video"].to(device)
        y = batch["label"]
        out = model(x)
        pred = out["logits"].argmax(dim=1).cpu()
        preds.extend(pred.tolist())
        targets.extend(y.tolist())
    # 5 类 macro-F1（用于 best 判断）+ 二分类 fall vs non-fall（对比用）
    metrics = compute_multiclass_metrics(preds, targets, num_classes=num_classes)
    binary = compute_fall_detection_metrics(preds, targets)
    metrics["binary_fall"] = binary
    print_metrics(metrics, "Validation")
    print(f"  [Binary fall vs non-fall] P={binary['precision']:.4f} "
          f"R={binary['recall']:.4f} F1={binary['f1']:.4f} "
          f"TP={binary['tp']} FP={binary['fp']} FN={binary['fn']}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["slowfast", "video_swin", "mvit"], required=True)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--resume", action="store_true",
                        help="从 teacher_<name>_last.pt 恢复训练")
    parser.add_argument("--resume_from", choices=["last", "best"], default="last",
                        help="恢复源: last=完整训练状态(含optimizer/scheduler), "
                             "best=从best_ckpt权重重启(不恢复optimizer/scheduler, 用于改loss后重训)")
    parser.add_argument("--class_weights", type=str, default=None,
                        help="5类权重, 逗号分隔, 如 '1.0,1.2,1.8,1.5,3.0' "
                             "(ADL,Fall,Fall-like,Lying,Transition). 默认无加权")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="CrossEntropy label smoothing, 0=关闭, 建议0.1")
    parser.add_argument("--ckpt_every", type=int, default=5,
                        help="每 N 个 epoch 存一次 last checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)

    # 模型
    model = build_teacher(args.teacher, num_classes=cfg.num_classes).to(device)
    print(f"[Teacher: {args.teacher}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # loss: 可选 class weight + label smoothing
    if args.class_weights:
        w = torch.tensor([float(v) for v in args.class_weights.split(",")],
                         device=device)
        assert w.numel() == cfg.num_classes, \
            f"class_weights 长度 {w.numel()} != num_classes {cfg.num_classes}"
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
        print(f"[Loss] weighted CE, weight={w.tolist()}, smoothing={args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        if args.label_smoothing > 0:
            print(f"[Loss] CE with label_smoothing={args.label_smoothing}")

    # 断点续传
    last_ckpt = os.path.join(args.save_dir, f"teacher_{args.teacher}_last.pt")
    best_ckpt = os.path.join(args.save_dir, f"teacher_{args.teacher}_best.pt")
    start_epoch = 0
    best_f1 = 0
    if args.resume:
        resume_path = best_ckpt if args.resume_from == "best" else last_ckpt
        if not os.path.exists(resume_path):
            print(f"[Resume] {resume_path} 不存在, 从头开始")
        else:
            print(f"[Resume] Loading {resume_path} (source={args.resume_from}) ...")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            # best 重启: 不恢复 optimizer/scheduler, 因为 loss 变了,
            # 旧 optimizer 动量会对新 loss landscape 失配
            if args.resume_from == "last":
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt.get("epoch", -1) + 1
            # best_f1 始终继承, 避免后续覆盖已有 best
            best_f1 = ckpt.get("best_f1", ckpt.get("f1", 0))
            print(f"[Resume] best_f1={best_f1:.4f}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device, args.teacher)
        scheduler.step()

        # 每 ckpt_every 个 epoch 存 last
        if (epoch + 1) % args.ckpt_every == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
            }, last_ckpt)
            print(f"  Checkpoint: {last_ckpt} (epoch {epoch+1})")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device, num_classes=cfg.num_classes)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                ckpt_path = os.path.join(args.save_dir, f"teacher_{args.teacher}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "f1": best_f1,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path} (macro_f1={best_f1:.4f})")


if __name__ == "__main__":
    main()
