"""架构验证脚本：无需训练即可验证代码可跑通

验证内容：
1. 学生模型前向跑通（参数统计）
2. 三教师前向跑通
3. 蒸馏损失计算跑通
4. 后处理流程跑通
5. 推理 demo 跑通（假数据）
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np

from configs.default import cfg
from models.student import build_student
from models.teachers import build_teacher
from utils.distill import DistillLoss
from utils.postprocess import FallAlarmPostprocess, Box
from utils.metrics import compute_fall_detection_metrics, print_metrics


def test_student_forward():
    print("\n===== Test 1: Student forward =====")
    student = build_student(cfg)
    stats = student.count_parameters()
    print(f"  Total params: {stats['total_M']}M (≤20M: {'OK' if stats['total']<=20e6 else 'FAIL'})")
    print(f"  FP32 size:   {stats['fp32_MB']}MB (≤80MB: {'OK' if stats['fp32_MB']<=80 else 'FAIL'})")
    print(f"    backbone: {stats['backbone']/1e6:.2f}M")
    print(f"    adapter:  {stats['adapter']/1e6:.2f}M")
    print(f"    head:     {stats['head']/1e6:.2f}M")
    x = torch.randn(2, cfg.seq_len, 3, *cfg.input_size)
    out = student(x)
    print(f"  Forward OK. logits={tuple(out['logits'].shape)} feat={tuple(out['feat'].shape)}")
    assert out["logits"].shape == (2, cfg.num_classes)
    assert out["feat"].shape[0] == 2
    return student


def test_teachers():
    print("\n===== Test 2: Teachers forward =====")
    x = torch.randn(2, cfg.seq_len, 3, *cfg.input_size)
    for name in ["slowfast", "video_swin", "mvit"]:
        t = build_teacher(name, num_classes=cfg.num_classes)
        out = t(x)
        n_param = sum(p.numel() for p in t.parameters())/1e6
        print(f"  {name}: params={n_param:.2f}M, logits={tuple(out['logits'].shape)} OK")


def test_distill_loss():
    print("\n===== Test 3: Distill loss =====")
    student = build_student(cfg)
    teachers = {n: build_teacher(n, cfg.num_classes) for n in ["slowfast", "video_swin", "mvit"]}
    loss_fn = DistillLoss(cfg)
    x = torch.randn(2, cfg.seq_len, 3, *cfg.input_size)
    y = torch.tensor([1, 0])
    s_out = student(x)
    t_outs = {n: t(x) for n, t in teachers.items()}
    loss = loss_fn(s_out, t_outs, y)
    print(f"  Loss components:")
    for k, v in loss.items():
        if k == "total":
            print(f"    total: {v.item():.4f}")
            v.backward()
            print(f"    backward OK")
        else:
            print(f"    {k}: {v if isinstance(v, float) else v.item():.4f}")


def test_postprocess():
    print("\n===== Test 4: Postprocess =====")
    postprocess = FallAlarmPostprocess()
    # 模拟跌倒序列
    for i in range(50):
        p_fall = 0.9 if 10 <= i <= 40 else 0.1
        box = Box(0, 100 + i, 100, 200 + i) if i < 20 else Box(0, 100, 100, 50)
        alarm = postprocess.update(p_fall, [box], i)
        if alarm:
            print(f"  Alarm triggered at frame {i}")


def test_metrics():
    print("\n===== Test 5: Metrics =====")
    preds = [0, 1, 1, 2, 0, 3, 1, 0]
    targets = [0, 1, 0, 2, 0, 3, 1, 1]
    metrics = compute_fall_detection_metrics(preds, targets)
    print_metrics(metrics, "Demo")


def test_param_constraint():
    print("\n===== Test 6: Hard constraints =====")
    student = build_student(cfg)
    stats = student.count_parameters()

    # 参数 ≤20M
    assert stats["total"] <= 20e6, f"Params {stats['total']} > 20M"
    print(f"  [OK] params {stats['total_M']}M ≤ 20M")

    # fp32 ≤80MB
    assert stats["fp32_MB"] <= 80, f"FP32 {stats['fp32_MB']} > 80MB"
    print(f"  [OK] fp32 {stats['fp32_MB']}MB ≤ 80MB")

    # INT8 大小估算 ≤20MB
    int8_mb = stats["total"] * 1 / 1e6  # INT8 每参数 1 byte
    assert int8_mb <= 20, f"INT8 {int8_mb} > 20MB"
    print(f"  [OK] INT8 size ~{int8_mb:.2f}MB ≤ 20MB")

    print(f"\n[ALL PASS] Hard constraints all satisfied.")


def main():
    print("=" * 60)
    print("Architecture verification")
    print("=" * 60)

    test_student_forward()
    test_teachers()
    test_distill_loss()
    test_postprocess()
    test_metrics()
    test_param_constraint()

    print("\n" + "=" * 60)
    print("All tests passed. Code is runnable.")
    print("=" * 60)


if __name__ == "__main__":
    main()
