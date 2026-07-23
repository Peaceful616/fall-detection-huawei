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
        # 特征对齐：初始化为 None，forward 中根据实际维度动态创建
        self.feat_align_slowfast = None
        self.feat_align_video_swin = None

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
        loss_feat = torch.tensor(0.0, device=labels.device)
        if "slowfast" in teacher_outs:
            # 教师 feat_list[0] 是 (B, C_t) 全局特征向量
            t_feat = teacher_outs["slowfast"]["feat_list"][0]  # (B, C_t)
            s_feat = student_out["backbone_feat"]  # (B, T, C, H, W)
            # 学生特征转成 (B, C, T, H, W) 然后 GAP 到 (B, C)
            s_feat_5d = s_feat.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
            s_feat_vec = F.adaptive_avg_pool3d(s_feat_5d, 1).flatten(1)  # (B, C)
            # 通道对齐
            if self.feat_align_slowfast is None or s_feat_vec.size(1) != self.feat_align_slowfast.out_channels:
                self.feat_align_slowfast = nn.Linear(t_feat.size(1), s_feat_vec.size(1)).to(labels.device)
            t_feat_aligned = self.feat_align_slowfast(t_feat)
            loss_feat = loss_feat + F.mse_loss(s_feat_vec, t_feat_aligned)
            loss_feat = loss_feat + F.mse_loss(F.normalize(s_feat_vec, dim=1), F.normalize(t_feat_aligned, dim=1))

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
        loss_rkd = torch.tensor(0.0, device=labels.device)
        if "video_swin" in teacher_outs:
            s_feat = student_out["feat"]  # (B, C, T)
            s_feat_flat = s_feat.mean(dim=2)  # (B, C)
            t_feat = teacher_outs["video_swin"]["feat_list"][0]  # (B, C_t)
            if self.feat_align_video_swin is None or s_feat_flat.size(1) != self.feat_align_video_swin.out_channels:
                self.feat_align_video_swin = nn.Linear(t_feat.size(1), s_feat_flat.size(1)).to(labels.device)
            t_feat_aligned = self.feat_align_video_swin(t_feat)
            loss_rkd = rkd_loss(s_feat_flat, t_feat_aligned)

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
