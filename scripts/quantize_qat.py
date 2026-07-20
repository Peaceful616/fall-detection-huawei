"""INT8 量化感知训练（QAT）

在学生蒸馏训练完成后做 QAT，保证量化后精度不崩
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
from models.student import build_student
from data.dataset import build_datasets
from utils.metrics import compute_fall_detection_metrics, print_metrics


def prepare_qat(model: nn.Module) -> nn.Module:
    """QAT 准备：插入伪量化节点"""
    # PyTorch 内置 QAT 配置
    qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model.qconfig = qconfig
    # 对每个子模块设置（更细粒度可改）
    torch.quantization.prepare_qat(model, inplace=True)
    return model


def convert_quantized(model: nn.Module) -> nn.Module:
    """转换为 INT8 量化模型"""
    model.eval()
    model_cpu = model.cpu()
    torch.quantization.convert(model_cpu, inplace=True)
    return model_cpu


def calibrate(model, loader, device, max_batches=20):
    """校准（observer 收集统计量）"""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch["video"].to(device)
            model(x)
            print(f"  Calibrate batch {i+1}/{max_batches}", end='\r')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="蒸馏后的学生权重 student_best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save", default=os.path.join(ROOT, "checkpoints", "student_int8.pt"))
    parser.add_argument("--export_onnx", default=os.path.join(ROOT, "checkpoints", "student_int8.onnx"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载蒸馏后的模型
    student = build_student(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[OK] Loaded student ckpt: {args.ckpt}")

    # 2. 准备 QAT
    student_qat = prepare_qat(student)
    print("[OK] QAT prepared")

    # 3. 短期微调（让 BN 与量化噪声对齐）
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(student_qat.parameters(), lr=cfg.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    student_qat.train()
    for epoch in range(args.epochs):
        total_loss, n = 0, 0
        for batch in train_loader:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            out = student_qat(x)
            loss = criterion(out["logits"], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  [QAT Epoch {epoch+1}] loss={total_loss/n:.4f}")

    # 4. 校准
    calibrate(student_qat, val_loader, device, max_batches=20)

    # 5. 转换为 INT8
    student_int8 = convert_quantized(student_qat)
    print("[OK] Converted to INT8")

    # 6. 保存
    torch.save(student_int8.state_dict(), args.save)
    print(f"[OK] Saved INT8 model: {args.save}")

    # 7. 导出 ONNX
    student_int8.eval()
    dummy = torch.randn(1, cfg.seq_len, 3, *cfg.input_size)
    torch.onnx.export(
        student_int8, dummy, args.export_onnx,
        input_names=["input"], output_names=["logits"],
        opset_version=13
    )
    print(f"[OK] Exported ONNX: {args.export_onnx}")

    # 8. 参数统计（INT8 vs fp32）
    n_int8 = sum(p.numel() for p in student_int8.parameters())
    n_int8_bytes = sum(p.numel() * p.element_size() for p in student_int8.parameters())
    print(f"[Stat] INT8 params: {n_int8/1e6:.2f}M, total size: {n_int8_bytes/1e6:.2f}MB")


if __name__ == "__main__":
    main()
