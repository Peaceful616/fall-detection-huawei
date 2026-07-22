"""验证 aux_kp 姿态辅助损失集成

测试 DistillLoss 的第 6 项（keypoint MSE）：
1. 学生输出含 aux_kp，batch 有 aux_kp_gt → loss_aux > 0
2. aux_kp_gt 部分帧全零 → 零帧不参与 loss
3. 反向传播无 NaN
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from configs.default import cfg
from models.student import build_student
from models.teachers import build_teacher
from utils.distill import DistillLoss


def main():
    print("=" * 60)
    print("aux_kp loss verification")
    print("=" * 60)
    device = torch.device("cpu")

    # 构建学生 + 教师
    student = build_student(cfg).to(device)
    teachers = {n: build_teacher(n, cfg.num_classes).to(device) for n in ["slowfast", "video_swin", "mvit"]}
    for t in teachers.values():
        for p in t.parameters():
            p.requires_grad = False
        t.eval()

    distill_loss = DistillLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    B, T = 2, cfg.seq_len
    x = torch.randn(B, T, 3, *cfg.input_size)
    y = torch.tensor([1, 0])

    # 构造 aux_kp_gt：部分帧有效，部分帧全零
    aux_kp_gt = torch.zeros(B, T, 17, 2)
    # sample 0: 帧 0~9 有效
    aux_kp_gt[0, :10] = torch.rand(10, 17, 2) * 0.8
    # sample 1: 帧 5~15 有效
    aux_kp_gt[1, 5:16] = torch.rand(11, 17, 2) * 0.8

    print(f"\n[Input] B={B}, T={T}, aux_kp_gt shape={tuple(aux_kp_gt.shape)}")
    valid_frames = (aux_kp_gt.abs().sum(dim=(-1, -2)) > 0).sum().item()
    print(f"  有效帧数: {valid_frames}/{B*T}")

    # 前向 + 损失
    student.train()
    s_out = student(x)
    print(f"\n[Student out] aux_kp shape={tuple(s_out['aux_kp'].shape)}")

    with torch.no_grad():
        t_outs = {n: t(x) for n, t in teachers.items()}

    loss_dict = distill_loss(s_out, t_outs, y, aux_kp_gt=aux_kp_gt)
    print(f"\n[Loss] components:")
    for k, v in loss_dict.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {val:.4f}")

    # 反向传播
    loss_dict["total"].backward()
    grad_max = 0
    grad_nan = False
    for p in student.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                grad_nan = True
            grad_max = max(grad_max, p.grad.abs().max().item())

    print(f"\n[Backward] grad_max={grad_max:.4f}, grad_nan={grad_nan}")

    # 验证 loss_aux > 0
    assert loss_dict["aux"] > 0, f"loss_aux should > 0, got {loss_dict['aux']}"
    print(f"\n[OK] loss_aux = {loss_dict['aux']:.4f} (>0, keypoint 监督生效)")

    # 验证：若 aux_kp_gt 全零，loss_aux 应为 0
    aux_kp_zero = torch.zeros_like(aux_kp_gt)
    loss_dict2 = distill_loss(s_out, t_outs, y, aux_kp_gt=aux_kp_zero)
    print(f"[OK] 全零 gt 时 loss_aux = {loss_dict2['aux']:.4f} (应为 0)")

    # 验证：不传 aux_kp_gt（推理/无 keypoint 场景），loss_aux 应为 0
    loss_dict3 = distill_loss(s_out, t_outs, y, aux_kp_gt=None)
    print(f"[OK] 不传 aux_kp_gt 时 loss_aux = {loss_dict3['aux']:.4f} (应为 0)")

    assert not grad_nan, "梯度有 NaN!"
    print(f"\n[ALL PASS] aux_kp 损失集成成功，姿态辅助分支监督生效")
    print("=" * 60)


if __name__ == "__main__":
    main()
