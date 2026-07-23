"""探测 OmniFall 数据集真实字段结构

在远程执行：
    pip install omnifall datasets -q
    python scripts/probe_omnifall.py

输出：
1. of-sta-cs 的 split / 列名 / 第一条样本
2. 16 类 label 的取值分布
3. 视频路径字段格式（path / video / dataset 来源）
4. 是否含红外/夜视标记（看 metadata 列）
5. 保存一份 sample json 到 ./data/omnifall_probe/
"""
import argparse
import os
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./data/omnifall_probe")
    parser.add_argument("--config", default="of-sta-cs",
                        help="OmniFall config，默认 of-sta-cs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[FAIL] pip install datasets -q")
        return

    print(f"[INFO] load_dataset('simplexsigil2/omnifall', '{args.config}') ...")
    ds = load_dataset("simplexsigil2/omnifall", args.config)
    print(f"[OK] splits: {list(ds.keys())}")

    summary = {"config": args.config, "splits": {}}
    for split, d in ds.items():
        cols = d.column_names
        n = len(d)
        print(f"\n=== {split}: n={n} columns={cols} ===")

        # 第一条样本
        s0 = d[0]
        print("[INFO] first sample:")
        for k, v in s0.items():
            vstr = str(v)
            print(f"  {k}: {vstr[:300]}")

        # label 分布（尝试 label / labels / activity / class）
        label_key = None
        for cand in ["label", "labels", "activity", "class", "action"]:
            if cand in cols:
                label_key = cand
                break
        label_dist = {}
        if label_key:
            labels = [s[label_key] for s in d]
            label_dist = dict(Counter(labels))
            print(f"[INFO] {label_key} distribution: {label_dist}")

        # dataset/source 分布
        src_dist = {}
        for cand in ["dataset", "source", "src", "db", "origin"]:
            if cand in cols:
                srcs = [s[cand] for s in d]
                src_dist[cand] = dict(Counter(srcs))
                print(f"[INFO] '{cand}' distribution: {src_dist[cand]}")
                break

        # 红外/夜视相关列
        ir_cols = [c for c in cols if any(
            k in c.lower() for k in ['ir', 'thermal', 'night', 'modality', 'view', 'sensor']
        )]
        print(f"[INFO] possible IR/view/modality columns: {ir_cols}")
        for c in ir_cols[:3]:
            vals = Counter(s.get(c) for s in d)
            print(f"  {c}: {dict(list(vals.items())[:20])}")

        # 保存前 5 条样本
        sample_path = os.path.join(args.out, f"sample_{split}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            samples = [{k: (str(v)[:500] if not isinstance(v, (int, float, bool, type(None))) else v)
                         for k, v in s.items()} for s in [d[i] for i in range(min(5, n))]]
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[OK] saved 5 samples -> {sample_path}")

        summary["splits"][split] = {
            "n": n,
            "columns": cols,
            "label_key": label_key,
            "label_dist": label_dist,
            "src_dist": src_dist,
            "ir_cols": ir_cols,
        }

    # 保存 summary
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] summary -> {summary_path}")
    print("\n[NOTE] 把 summary.json 和 sample_*.json 的内容贴回来，"
          "我会据此写 omnifall_adapter.py 和 16 类 -> 5 类 label 映射。")


if __name__ == "__main__":
    main()
