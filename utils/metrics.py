"""评测指标"""
import numpy as np
import torch
from collections import defaultdict


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 准确率"""
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def precision_recall_f1(confusion: np.ndarray, num_classes: int = 5):
    """从混淆矩阵计算每类 P/R/F1

    confusion: (num_classes, num_classes)  [true, pred]
    """
    metrics = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        metrics.append({"class": c, "precision": p, "recall": r, "f1": f1})
    return metrics


def macro_f1(metrics: list) -> float:
    """宏平均 F1"""
    return np.mean([m["f1"] for m in metrics])


def build_confusion_matrix(preds: list, targets: list, num_classes: int = 5) -> np.ndarray:
    """构建混淆矩阵"""
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        mat[t, p] += 1
    return mat


def evaluate_scene_wise(preds: list, targets: list, scenes: list, num_classes: int = 5):
    """分场景评测"""
    scene_data = defaultdict(lambda: {"preds": [], "targets": []})
    for p, t, s in zip(preds, targets, scenes):
        scene_data[s]["preds"].append(p)
        scene_data[s]["targets"].append(t)
    out = {}
    for scene, d in scene_data.items():
        cm = build_confusion_matrix(d["preds"], d["targets"], num_classes)
        m = precision_recall_f1(cm, num_classes)
        out[scene] = {
            "count": len(d["preds"]),
            "f1": macro_f1(m),
            "metrics": m,
        }
    return out


def compute_fall_detection_metrics(preds: list, targets: list):
    """二分类跌倒检测指标（fall vs non-fall）

    fall = 1, fall-like = 2 → 视为 fall 正类
    ADL = 0, lying = 3, transition = 4 → 视为 non-fall
    """
    binary_preds = [1 if p in [1, 2] else 0 for p in preds]
    binary_targets = [1 if t in [1, 2] else 0 for t in targets]
    tp = sum(1 for p, t in zip(binary_preds, binary_targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(binary_preds, binary_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(binary_preds, binary_targets) if p == 0 and t == 1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn
    }


# 5 类名（与 dataset.py / configs 对齐）
CLASS5_NAMES = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]


def compute_multiclass_metrics(preds: list, targets: list,
                               num_classes: int = 5,
                               class_names=None):
    """多分类指标：每类 P/R/F1 + macro-F1 + 混淆矩阵

    用于 OmniFall 16→5 映射后的 5 类细判别评测。
    返回 dict：{
        "f1": macro_f1,              # 兼容 print_metrics 的 f1 字段
        "macro_f1": macro_f1,
        "per_class": [{class, precision, recall, f1, support}, ...],
        "confusion_matrix": np.ndarray (num_classes x num_classes),
        "accuracy": top1_acc,
    }
    """
    if class_names is None:
        class_names = CLASS5_NAMES[:num_classes]
    cm = build_confusion_matrix(preds, targets, num_classes=num_classes)
    per_class = []
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        support = int(cm[c, :].sum())
        per_class.append({
            "class_id": c,
            "class_name": class_names[c],
            "precision": p, "recall": r, "f1": f1,
            "support": support,
        })
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    total = cm.sum()
    correct = int(np.trace(cm))
    acc = correct / (total + 1e-8)
    return {
        "f1": macro_f1,  # 兼容旧代码用 metrics["f1"]
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: dict, title: str = ""):
    """打印指标"""
    print(f"\n===== {title} =====")
    if "macro_f1" in metrics:
        # 多分类
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Per-class (P/R/F1, support):")
        for c in metrics.get("per_class", []):
            print(f"  [{c['class_name']}] P={c['precision']:.4f} "
                  f"R={c['recall']:.4f} F1={c['f1']:.4f} (n={c['support']})")
        cm = metrics.get("confusion_matrix")
        if cm:
            names = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]
            print(f"Confusion matrix (rows=true, cols=pred):")
            print(f"            " + "  ".join(f"{n[:6]:>7s}" for n in names[:len(cm)]))
            for i, row in enumerate(cm):
                print(f"  {names[i][:7]:<7s} " + "  ".join(f"{v:7d}" for v in row))
    elif "precision" in metrics and "recall" in metrics:
        # 二分类
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"TP={metrics.get('tp', 0)} FP={metrics.get('fp', 0)} FN={metrics.get('fn', 0)}")
    for k, v in metrics.items():
        if isinstance(v, dict) and "f1" in v and "precision" not in v:
            print(f"  [{k}] F1={v['f1']:.4f} (n={v.get('count', 0)})")
