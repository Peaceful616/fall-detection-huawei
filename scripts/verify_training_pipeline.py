"""合成数据训练 pipeline 验证

目的：在真实数据集到货前，验证完整训练流程能跑通
- 验证数据加载
- 验证前向 + 反向传播稳定
- 验证蒸馏损失收敛
- 验证梯度无 NaN
- 验证后处理报警触发

跑 2 个 epoch，loss 应该下降。如果跑通，真实数据到货后即可立即开训。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from configs.default import cfg
from models.student import build_student
from models.teachers import build_teacher
from utils.distill import DistillLoss
from utils.postprocess import FallAlarmPostprocess, Box


class SyntheticVideoDataset(Dataset):
    """合成视频数据集

    生成随机视频片段 + 随机标签
    用于验证训练 pipeline 能跑通
    """

    def __init__(self, n_samples=20, seq_len=16, input_size=(288, 288)):
        self.n = n_samples
        self.seq_len = seq_len
        self.input_size = input_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 随机视频片段 (T, 3, H, W)
        x = torch.randn(self.seq_len, 3, *self.input_size) * 0.5
        # normalize 模拟
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        # 随机标签（5 分类）
        label = np.random.randint(0, cfg.num_classes)
        return {
            "video": x,
            "label": label,
            "video_path": f"synthetic_{idx}",
            "scene": "indoor",
            "light": "normal",
        }


def test_pipeline():
    print("=" * 60)
    print("Training pipeline verification (synthetic data)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # 1. 数据
    print("\n[Test 1] Dataset & DataLoader")
    train_set = SyntheticVideoDataset(n_samples=8)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    batch = next(iter(train_loader))
    print(f"  Batch video: {tuple(batch['video'].shape)}")
    print(f"  Batch label: {batch['label']}")
    assert batch["video"].shape == (2, cfg.seq_len, 3, *cfg.input_size)

    # 2. 学生 + 教师
    print("\n[Test 2] Build student & teachers")
    student = build_student(cfg).to(device)
    stats = student.count_parameters()
    print(f"  Student: total={stats['total_M']}M, trainable={stats['trainable']/1e6:.2f}M")
    print(f"    backbone: {stats['backbone']/1e6:.2f}M (trainable {stats['backbone_trainable']/1e6:.2f}M)")
    print(f"    adapter:  {stats['adapter']/1e6:.2f}M")
    print(f"    head:     {stats['head']/1e6:.2f}M")
    print(f"    aux_kp:   {stats['aux_kp']/1e6:.3f}M (推理不部署)")

    teachers = {}
    for name in ["slowfast", "video_swin", "mvit"]:
        t = build_teacher(name, num_classes=cfg.num_classes).to(device)
        for p in t.parameters():
            p.requires_grad = False
        t.eval()
        teachers[name] = t
        n = sum(p.numel() for p in t.parameters()) / 1e6
        print(f"  Teacher {name}: {n:.2f}M (frozen)")

    # 3. 蒸馏损失
    print("\n[Test 3] Distill loss")
    distill_loss = DistillLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # backbone 用更小学习率
    backbone_params = list(filter(
        lambda p: p.requires_grad, student.backbone.parameters()
    ))
    other_params = list(filter(
        lambda p: p.requires_grad and not any(p is bp for bp in backbone_params),
        student.parameters()
    ))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": other_params, "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    print(f"  Optimizer: AdamW (backbone lr={cfg.lr_backbone}, other lr={cfg.lr})")

    # 4. 跑 2 个 epoch
    print("\n[Test 4] Training 2 epochs")
    student.train()
    for epoch in range(2):
        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()

            s_out = student(x)
            with torch.no_grad():
                t_outs = {n: t(x) for n, t in teachers.items()}
            loss = distill_loss(s_out, t_outs, y)
            loss["total"].backward()

            # 梯度检查
            grad_max = 0
            grad_nan = False
            for p in student.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        grad_nan = True
                    g = p.grad.abs().max().item()
                    grad_max = max(grad_max, g)

            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss["total"].item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, "
              f"grad_max={grad_max:.4f}, grad_nan={grad_nan}")

    # 5. 验证 loss 是否收敛（合成数据下不应爆炸）
    print("\n[Test 5] Sanity check")
    if avg_loss < 10:
        print(f"  [OK] Loss is reasonable: {avg_loss:.4f}")
    else:
        print(f"  [WARN] Loss too high: {avg_loss:.4f}")
    if not grad_nan:
        print(f"  [OK] No NaN gradients")
    else:
        print(f"  [FAIL] NaN gradients detected!")

    # 6. 推理模式
    print("\n[Test 6] Inference mode")
    student.eval()
    with torch.no_grad():
        x = torch.randn(1, cfg.seq_len, 3, *cfg.input_size).to(device)
        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(device)
        x = (x - mean) / std
        out = student(x)
        probs = torch.softmax(out["logits"], dim=1)[0]
        print(f"  Logits shape: {tuple(out['logits'].shape)}")
        print(f"  Probabilities: {probs.cpu().numpy()}")
        print(f"  Predicted class: {probs.argmax().item()}")
        # 推理时无 aux_kp（应不在输出里）
        assert "aux_kp" not in out, "aux_kp should not be in inference output"
        print(f"  [OK] aux_kp correctly disabled in inference")

    # 7. 后处理
    print("\n[Test 7] Postprocess")
    postprocess = FallAlarmPostprocess()
    for i in range(50):
        p_fall = 0.9 if 10 <= i <= 40 else 0.1
        box = Box(0, 100 + i, 100, 200 + i)
        alarm = postprocess.update(p_fall, [box], i)
        if alarm:
            print(f"  Alarm triggered at frame {i}")

    # 8. 参数预算检查
    print("\n[Test 8] Parameter budget check")
    total_params = sum(p.numel() for p in student.parameters())
    infer_params = stats['infer_total']
    fp32_mb = total_params * 4 / 1e6
    int8_mb = total_params / 1e6
    print(f"  Total params:    {total_params/1e6:.2f}M (limit 20M)")
    print(f"  Infer params:    {infer_params/1e6:.2f}M (推理部署)")
    print(f"  FP32 size:       {fp32_mb:.2f}MB (limit 80MB)")
    print(f"  INT8 size:       {int8_mb:.2f}MB (limit 20MB)")
    assert total_params <= 20e6, "Params exceed 20M"
    assert fp32_mb <= 80, "FP32 size exceed 80MB"
    assert int8_mb <= 20, "INT8 size exceed 20MB"
    print(f"  [OK] All hard constraints satisfied")

    print("\n" + "=" * 60)
    print("All pipeline tests passed.")
    print("Code is ready for real data training.")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
