"""统计 OF-Syn train 集 5 类样本分布, 用于算 class weight.

用法:
    python scripts/probe_class_dist.py [--ann data/omnifall_syn/annotations/annotations_train.json]

输出各类 support, 以及三种 weight 方案:
1. inverse (1/n_i): 经典逆频次
2. sqrt_inverse (1/sqrt(n_i)): 温和版, 防止少数类权重过大爆梯度
3. effective (有效样本数, Cui 2019): (1-beta)/(1-beta^n_i), beta=0.999
"""
import argparse
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/omnifall_syn/annotations/annotations_train.json")
    args = ap.parse_args()

    if not os.path.exists(args.ann):
        # 尝试其他可能路径
        candidates = [
            args.ann,
            "annotations_train.json",
            "data/annotations_train.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.ann = c
                break
        else:
            print(f"[ERROR] 找不到 {args.ann}")
            print("请用 --ann 指定 annotations_train.json 的路径")
            return

    with open(args.ann, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # 兼容 {samples: [...]} 或直接 list
    if isinstance(ann, dict) and "samples" in ann:
        samples = ann["samples"]
    elif isinstance(ann, list):
        samples = ann
    else:
        # 尝试常见的 key
        for k in ("data", "annotations", "train"):
            if k in ann:
                samples = ann[k]
                break
        else:
            print(f"[ERROR] 无法解析 annotations 结构, top-level keys: {list(ann.keys()) if isinstance(ann, dict) else type(ann)}")
            return

    labels = [int(s["label"]) for s in samples]
    cnt = Counter(labels)
    print(f"\n== OF-Syn train 集类别分布 ==")
    names = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]
    n_total = sum(cnt.values())
    for i in range(5):
        n = cnt.get(i, 0)
        print(f"  [{i}] {names[i]:12s} n={n:5d}  ({100*n/n_total:.1f}%)")
    print(f"  total = {n_total}")

    # 方案 1: inverse
    print(f"\n== 方案1: inverse (1/n) ==")
    w1 = [n_total / (5.0 * cnt.get(i, 1)) for i in range(5)]
    # 归一化到最小类=1, 便于和手工值对比
    w1_norm = [x / min(w1) for x in w1]
    print(f"  raw:       {[f'{x:.4f}' for x in w1]}")
    print(f"  normalized (min=1): {[f'{x:.3f}' for x in w1_norm]}")
    print(f"  逗号分隔: {','.join(f'{x:.3f}' for x in w1_norm)}")

    # 方案 2: sqrt inverse (温和)
    print(f"\n== 方案2: sqrt inverse (1/sqrt(n)) - 推荐 ==")
    w2 = [(n_total / (5.0 * cnt.get(i, 1))) ** 0.5 for i in range(5)]
    w2_norm = [x / min(w2) for x in w2]
    print(f"  raw:       {[f'{x:.4f}' for x in w2]}")
    print(f"  normalized (min=1): {[f'{x:.3f}' for x in w2_norm]}")
    print(f"  逗号分隔: {','.join(f'{x:.3f}' for x in w2_norm)}")

    # 方案 3: effective number of samples (Cui 2019)
    print(f"\n== 方案3: effective number (Cui 2019, beta=0.999) ==")
    beta = 0.999
    w3 = []
    for i in range(5):
        n = cnt.get(i, 1)
        eff = (1 - beta) / (1 - beta ** n)
        w3.append(1.0 / eff)
    w3_norm = [x / min(w3) for x in w3]
    print(f"  raw:       {[f'{x:.6f}' for x in w3]}")
    print(f"  normalized (min=1): {[f'{x:.3f}' for x in w3_norm]}")
    print(f"  逗号分隔: {','.join(f'{x:.3f}' for x in w3_norm)}")

    print(f"\n== 推荐 ==")
    print("用方案2 (sqrt inverse), 既纠正长尾又不会让 Transition 权重过大爆梯度。")
    print("方案1 inverse 太激进, 少数类权重可能是多数类的 5-10 倍, 训练不稳。")
    print("方案3 effective 更平滑, 但需要长训练才显效, 短 fine-tune 优势不明显。")


if __name__ == "__main__":
    main()
