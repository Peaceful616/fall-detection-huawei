"""多教师三重蒸馏损失

三重蒸馏：
1. 特征级：AT (Attention Transfer) + L2 对齐
2. logit 级：加权 KL 散度（温度 T=4）
3. 关系级：RKD 保持样本间相似度

额外：
4. 跨模态蒸馏：MViT 教师模态不变特征对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def attn_transfer_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """AT (Attention Transfer) 损失

    对齐 student 与 teacher 的注意力图（沿通道维平方和归一化）
    """
    # 归一化注意力图
    def attn_map(f):
        # f: (B, C, ...)
        return F.normalize(f.pow(2).mean(dim=1).flatten(1), dim=1)
    s_a = attn_map(student_feat)
    t_a = attn_map(teacher_feat)
    # 维度对齐（如果 T 与 S 的空间尺寸不同）
    if s_a.shape != t_a.shape:
        t_a = F.interpolate(t_a.unsqueeze(1), size=s_a.shape[1:], mode='linear').squeeze(1)
    return F.mse_loss(s_a, t_a)


def feat_l2_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                 align: nn.Module = None) -> torch.Tensor:
    """特征 L2 对齐损失

    若通道数不同，用 align (1x1 conv) 对齐
    """
    if align is not None:
        # (B, C_t, ...) → (B, C_s, ...)
        teacher_feat = align(teacher_feat)
    # 空间尺寸对齐
    if student_feat.shape[2:] != teacher_feat.shape[2:]:
        teacher_feat = F.interpolate(
            teacher_feat, size=student_feat.shape[2:], mode='bilinear', align_corners=False
        )
    return F.mse_loss(student_feat, teacher_feat.detach())


def logit_kd_loss(student_logits: torch.Tensor, teacher_logits_list: list,
                  temperature: float = 4.0, weights: list = None) -> torch.Tensor:
    """logit 蒸馏：多教师加权 KL 散度

    student_logits: (B, C)
    teacher_logits_list: List[(B, C)]
    """
    if weights is None:
        weights = [1.0 / len(teacher_logits_list)] * len(teacher_logits_list)
    # 加权融合的教师软标签
    soft_teacher = sum(
        w * F.softmax(t / temperature, dim=1)
        for w, t in zip(weights, teacher_logits_list)
    )
    soft_teacher = soft_teacher / soft_teacher.sum(dim=1, keepdim=True)
    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        soft_teacher, reduction='batchmean'
    ) * (temperature ** 2)
    return loss


def rkd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """关系蒸馏（RKD）

    保持样本间距离关系
    """
    # 展平为 (B, D)
    s = student_feat.flatten(1)
    t = teacher_feat.flatten(1).detach()
    # 维度对齐
    if s.size(1) != t.size(1):
        t = F.adaptive_avg_pool1d(t.unsqueeze(1), s.size(1)).squeeze(1)
    # 距离矩阵
    def pairwise_dist(f):
        n = f.size(0)
        d = torch.cdist(f, f, p=2)
        d = d / (d.max() + 1e-6)  # 归一化
        return d
    s_d = pairwise_dist(s)
    t_d = pairwise_dist(t)
    return F.smooth_l1_loss(s_d, t_d)


def modal_distill_loss(student_logits: torch.Tensor, teacher_modal_logits: torch.Tensor,
                       teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """跨模态蒸馏损失

    student 对齐 teacher 的主任务输出
    teacher 的 modal_logits 用于对抗，学生不直接用
    """
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)


class DistillLoss(nn.Module):
    """总蒸馏损失"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()
        # 特征对齐：从教师通道数到学生 backbone_feat 通道数（YOLOv8s P4 = 256）
        student_backbone_c = cfg.adapter_c_in  # 256 (YOLOv8s P4)
        # SlowFast 教师 feat 通道 256, 学生 256
        self.feat_align_slowfast = nn.Conv3d(256, student_backbone_c, 1)
        # VideoSwin 教师 feat 通道 96, 学生 adapter_c_out = 64
        self.feat_align_video_swin = nn.Linear(96, cfg.adapter_c_out)

    def forward(self, student_out: dict, teacher_outs: dict, labels: torch.Tensor,
               alpha_modal: float = None, aux_kp_gt: torch.Tensor = None):
        """
        student_out: {logits, feat, backbone_feat, aux_kp?}
        teacher_outs: {
            'slowfast': {logits, feat_list},
            'video_swin': {logits, feat_list},
            'mvit': {logits, feat_list, modal_logits}
        }
        labels: (B,)
        aux_kp_gt: (B, T, 17, 2) 真值关键点（归一化坐标），仅训练期提供
                  部分帧可能全零（CSV 未覆盖），用 mask 过滤
        """
        cfg = self.cfg
        if alpha_modal is None:
            alpha_modal = cfg.alpha_modal

        # 1. 主任务 CE
        loss_ce = self.ce(student_out["logits"], labels)

        # 2. 特征蒸馏（AT + L2）
        # 当前官方预训练教师只返回 logits，无法直接提取中间层 3D 特征，
        # 故暂时禁用特征蒸馏，后续可通过替换 backbone 获取中间特征。
        loss_feat = torch.tensor(0.0, device=labels.device)

        # 3. logit 蒸馏
        loss_logit = torch.tensor(0.0, device=labels.device)
        teacher_logits_list = []
        weights = []
        for name, w in cfg.teacher_weights.items():
            if name in teacher_outs:
                teacher_logits_list.append(teacher_outs[name]["logits"])
                weights.append(w)
        if teacher_logits_list:
            loss_logit = logit_kd_loss(
                student_out["logits"], teacher_logits_list,
                temperature=cfg.distill_temperature, weights=weights
            )

        # 4. 关系蒸馏
        # 当前官方预训练教师只返回 logits，无法直接提取特征向量，
        # 故暂时禁用关系蒸馏。
        loss_rkd = torch.tensor(0.0, device=labels.device)

        # 5. 跨模态蒸馏
        loss_modal = torch.tensor(0.0, device=labels.device)
        if "mvit" in teacher_outs:
            loss_modal = modal_distill_loss(
                student_out["logits"], teacher_outs["mvit"].get("modal_logits"),
                teacher_outs["mvit"]["logits"], cfg.distill_temperature
            )

        # 6. 姿态辅助监督（关键创新 4：让 backbone 学人体结构特征）
        # 学生 aux_kp: (B, T, 17, 2)，gt: (B, T, 17, 2)
        # CSV 部分帧未覆盖（全零），用 mask 过滤这些帧
        loss_aux = torch.tensor(0.0, device=labels.device)
        if (cfg.aux_kp_enabled and aux_kp_gt is not None
                and "aux_kp" in student_out):
            pred_kp = student_out["aux_kp"]  # (B, T, 17, 2)
            # mask: 有效帧 = gt 非全零 (B, T) - 按帧判定
            valid_mask = (aux_kp_gt.abs().sum(dim=(-1, -2)) > 0).float()  # (B, T)
            if valid_mask.sum() > 0:
                # 逐元素 MSE 仅在有效位置计算
                diff = (pred_kp - aux_kp_gt) ** 2  # (B, T, 17, 2)
                diff = diff.sum(dim=(-1, -2))  # (B, T)
                loss_aux = (diff * valid_mask).sum() / (valid_mask.sum() * 17 * 2 + 1e-8)

        total = (loss_ce
                 + cfg.alpha_feat * loss_feat
                 + cfg.alpha_logit * loss_logit
                 + cfg.alpha_rkd * loss_rkd
                 + alpha_modal * loss_modal
                 + cfg.alpha_aux * loss_aux)

        return {
            "total": total,
            "ce": loss_ce.item(),
            "feat": loss_feat.item() if isinstance(loss_feat, torch.Tensor) else loss_feat,
            "logit": loss_logit.item() if isinstance(loss_logit, torch.Tensor) else loss_logit,
            "rkd": loss_rkd.item() if isinstance(loss_rkd, torch.Tensor) else loss_rkd,
            "modal": loss_modal.item() if isinstance(loss_modal, torch.Tensor) else loss_modal,
            "aux": loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux,
        }
