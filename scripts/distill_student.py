"""学生模型蒸馏训练脚本

三教师冻结，学生用 DistillLoss 做三重蒸馏 + 跨模态
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.default import cfg
from models.student import build_student
from models.teachers import build_teacher
from data.dataset import build_datasets
from utils.distill import DistillLoss
from utils.metrics import accuracy, compute_fall_detection_metrics, print_metrics


def load_teacher_ckpt(name: str, model: nn.Module, ckpt_dir: str) -> bool:
    """加载教师权重，失败返回 False"""
    path = os.path.join(ckpt_dir, f"teacher_{name}_best.pt")
    if not os.path.exists(path):
        print(f"[WARN] Teacher ckpt not found: {path}, use random init")
        return False
    ckpt = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[OK] Loaded teacher {name} (F1={ckpt.get('f1', 0):.4f})")
        return True
    except Exception as e:
        print(f"[WARN] Failed to load teacher {name}: {e}")
        return False


def train_one_epoch(student, teachers, loader, optimizer, distill_loss, device, epoch):
    student.train()
    for t in teachers.values():
        t.eval()

    total = {"total": 0, "ce": 0, "feat": 0, "logit": 0, "rkd": 0, "modal": 0}
    n = 0
    for batch in loader:
        x = batch["video"].to(device)  # (B, T, 3, H, W)
        y = batch["label"].to(device)
        optimizer.zero_grad()

        # 学生前向
        s_out = student(x)

        # 教师前向（no_grad）
        t_outs = {}
        with torch.no_grad():
            for name, t in teachers.items():
                t_out = t(x)
                t_outs[name] = t_out

        # 蒸馏损失
        loss_dict = distill_loss(s_out, t_outs, y)
        loss_dict["total"].backward()
        optimizer.step()

        for k in total:
            total[k] += loss_dict[k] * x.size(0)
        n += x.size(0)

    msg = "  ".join(f"{k}={v/n:.4f}" for k, v in total.items())
    print(f"  [Distill Epoch {epoch+1}] {msg}")


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    preds, targets = [], []
    for batch in loader:
        x = batch["video"].to(device)
        y = batch["label"]
        out = student(x)
        pred = out["logits"].argmax(dim=1).cpu()
        preds.extend(pred.tolist())
        targets.extend(y.tolist())
    metrics = compute_fall_detection_metrics(preds, targets)
    print_metrics(metrics, "Student Validation")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--teacher_dir", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--teachers", nargs="+",
                       default=["slowfast", "video_swin", "mvit"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)

    # 学生
    student = build_student(cfg).to(device)
    stats = student.count_parameters()
    print(f"[Student] total={stats['total_M']}M fp32={stats['fp32_MB']}MB "
          f"backbone={stats['backbone']/1e6:.2f}M adapter={stats['adapter']/1e6:.2f}M "
          f"head={stats['head']/1e6:.2f}M")

    # 教师（冻结）
    teachers = {}
    for name in args.teachers:
        t = build_teacher(name, num_classes=cfg.num_classes).to(device)
        load_teacher_ckpt(name, t, args.teacher_dir)
        for p in t.parameters():
            p.requires_grad = False
        t.eval()
        teachers[name] = t

    # 蒸馏损失
    distill_loss = DistillLoss(cfg).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(student, teachers, train_loader, optimizer, distill_loss, device, epoch)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(student, val_loader, device)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                ckpt = os.path.join(args.save_dir, "student_best.pt")
                torch.save({
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "f1": best_f1,
                    "config": cfg.__dict__,
                }, ckpt)
                print(f"  Saved: {ckpt} (F1={best_f1:.4f})")

    print(f"\n[Done] Best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
