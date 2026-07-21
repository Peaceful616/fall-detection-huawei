"""YOLO 风格检测损失

损失组成：
- bbox loss: CIoU 回归
- objectness loss: BCE
- classification loss: CE

简化版：
- 每个样本有 N 个 GT box
- 每张图有 num_anchors 个 anchor 预测
- 匹配：anchor 与 GT 的 IoU 最大者负责该 GT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CIoU 损失

    pred: (N, 4) [cx, cy, w, h] 归一化
    target: (N, 4)
    """
    # 转为 (x1, y1, x2, y2)
    px1 = pred[:, 0] - pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2
    py2 = pred[:, 1] + pred[:, 3] / 2

    tx1 = target[:, 0] - target[:, 2] / 2
    ty1 = target[:, 1] - target[:, 3] / 2
    tx2 = target[:, 0] + target[:, 2] / 2
    ty2 = target[:, 1] + target[:, 3] / 2

    # 交集
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # 并集
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter
    iou = inter / (union + 1e-7)

    # 外接框
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    ew = ex2 - ex1
    eh = ey2 - ey1

    # 中心距离
    pdx = (target[:, 0] - pred[:, 0])
    pdy = (target[:, 1] - pred[:, 1])
    rho = pdx * pdx + pdy * pdy

    # 对角线平方
    c2 = ew * ew + eh * eh

    # 宽高比项（简化版）
    v = 0.0  # 完整 CIoU 还有 v 和 alpha，简化省略
    return 1 - iou + rho / (c2 + 1e-7)  # DIoU 简化版


class DetectionLoss(nn.Module):
    """YOLO 风格检测损失"""

    def __init__(self, num_anchors: int = 3, num_classes: int = 1,
                 lambda_box: float = 5.0, lambda_obj: float = 1.0,
                 lambda_cls: float = 1.0):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions: dict, targets: dict):
        """
        predictions: {bbox (B, A, 4), cls (B, A, C), obj (B, A)}
        targets: {boxes (B, N_max, 5), n_boxes (B,)}
        """
        bbox_pred = predictions["bbox"]  # (B, A, 4)
        cls_pred = predictions["cls"]    # (B, A, C)
        obj_pred = predictions["obj"]    # (B, A)

        target_boxes = targets["boxes"][:, :, 1:]  # (B, N, 4) 去掉 class_id
        target_cls = targets["boxes"][:, :, 0].long()  # (B, N)
        n_boxes = targets["n_boxes"]  # (B,)

        B, A, _ = bbox_pred.shape
        device = bbox_pred.device

        total_box_loss = 0
        total_obj_loss = 0
        total_cls_loss = 0

        for b in range(B):
            n = int(n_boxes[b].item())
            if n == 0:
                # 无目标：obj 全部应为 0
                obj_target = torch.zeros(A, device=device)
                total_obj_loss += self.bce(obj_pred[b], obj_target).mean()
                continue

            # 对每个 GT，找最匹配的 anchor（这里简化为"贪心分配"）
            gt = target_boxes[b, :n]  # (n, 4)
            # 计算每个 anchor 与每个 GT 的"距离"（用中心点距离简化）
            anchor_centers = bbox_pred[b, :, :2].detach()  # (A, 2)
            gt_centers = gt[:, :2]  # (n, 2)
            # 计算距离
            dist = torch.cdist(anchor_centers, gt_centers)  # (A, n)
            # 每个 GT 找距离最小的 anchor
            assigned = dist.argmin(dim=0)  # (n,)

            # 标记 anchor 是否被分配
            obj_target = torch.zeros(A, device=device)
            cls_target = torch.zeros(A, dtype=torch.long, device=device)
            box_target = torch.zeros(A, 4, device=device)

            for i in range(n):
                a = assigned[i].item()
                obj_target[a] = 1
                cls_target[a] = target_cls[b, i].item()
                box_target[a] = gt[i]

            # box loss
            matched_mask = obj_target.bool()
            if matched_mask.any():
                box_loss = ciou_loss(
                    bbox_pred[b][matched_mask],
                    box_target[matched_mask]
                ).mean()
                total_box_loss += box_loss

            # obj loss
            total_obj_loss += self.bce(obj_pred[b], obj_target).mean()

            # cls loss（仅对正样本）
            if matched_mask.any():
                cls_loss = F.cross_entropy(
                    cls_pred[b][matched_mask],
                    cls_target[matched_mask]
                )
                total_cls_loss += cls_loss

        loss = (self.lambda_box * total_box_loss / B +
                self.lambda_obj * total_obj_loss / B +
                self.lambda_cls * total_cls_loss / B)

        return {
            "total": loss,
            "box": (total_box_loss / max(B, 1)).item(),
            "obj": (total_obj_loss / max(B, 1)).item(),
            "cls": (total_cls_loss / max(B, 1)).item(),
        }
