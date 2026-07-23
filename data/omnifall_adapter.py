"""OmniFall 数据集适配器

数据集来源：HuggingFace simplexsigil2/omnifall
- 16 类活动 taxonomy
- 三大组件：OF-Staged (8 公开 lab 数据集) / OF-ItW (OOPS 真实意外) / OF-Syn (12000 合成视频)
- 70+ 配置，关键：
    of-sta-cs          8 staged 数据集 cross-subject 划分
    of-sta-cv          cross-view 划分
    of-sta-to-all-cs   训练 staged 测试 staged+itw+syn（cross-domain 泛化）

样本字段（README 示例）：
    {'path': ..., 'label': 1, 'start': 0.0, 'end': 2.5, ...}

本适配器：
1. 用 datasets.load_dataset 拉取指定 config
2. 把 16 类 label 映射到本项目的 5 类（Fall / Fall-like / ADL / Lying / Transition）
3. 导出 annotations_{train,val,test}.json 到 out_dir/annotations/
   字段与 kaggle_fall_adapter 对齐：
   {video_path(绝对), label(int 0-4), start, end, scene, light,
    source(子数据集名), is_ir(布尔), original_label(int 0-15)}
4. 视频路径直接用 OmniFall 返回的 path 字段（已是绝对路径或 HF cache 路径）

注意：本适配器只导出标注，不解码视频。视频读取仍走 VideoFallDataset
（cv2.VideoCapture）或先用 predecode_videos.py 预解码再用 Fast 版本。

用法：
    python data/omnifall_adapter.py --config of-sta-cs --out ./data/omnifall

16 类 taxonomy 映射策略（待 probe 确认 label 定义后调整）：
OmniFall 16 类（基于论文 arXiv:2505.19889 推断）：
    0  Fall                    -> 1 (Fall)
    1  Fall-like (stumble)     -> 2 (Fall-like)
    2  Lying                   -> 3 (Lying)
    3  Lying-like (floor)      -> 3 (Lying)
    4  Sit                     -> 0 (ADL)
    5  Sit-like (chair)        -> 0 (ADL)
    6  Bend                    -> 2 (Fall-like，弯腰易误判)
    7  Squat                    -> 2 (Fall-like，下蹲易误判)
    8  Walk                     -> 0 (ADL)
    9  Stand/Stand-up           -> 4 (Transition)
    10 Lie down (intentional)  -> 3 (Lying)
    11 Sit down                -> 4 (Transition)
    12 Stand up                -> 4 (Transition)
    13 Crouch                  -> 2 (Fall-like)
    14 Kneel                    -> 0 (ADL)
    15 Other ADL                -> 0 (ADL)

以上映射是初始猜测，probe 跑完拿到 LABELS.md 真实定义后会修正。
"""
import argparse
import os
import json
from collections import Counter
from pathlib import Path


# 16 类 -> 5 类映射（初始版本，待 probe 修正）
OMNIFALL16_TO_5 = {
    0: 1,    # Fall -> Fall
    1: 2,    # Fall-like (stumble) -> Fall-like
    2: 3,    # Lying -> Lying
    3: 3,    # Lying-like -> Lying
    4: 0,    # Sit -> ADL
    5: 0,    # Sit-like -> ADL
    6: 2,    # Bend -> Fall-like
    7: 2,    # Squat -> Fall-like
    8: 0,    # Walk -> ADL
    9: 4,    # Stand -> Transition
    10: 3,   # Lie down -> Lying
    11: 4,   # Sit down -> Transition
    12: 4,   # Stand up -> Transition
    13: 2,   # Crouch -> Fall-like
    14: 0,   # Kneel -> ADL
    15: 0,   # Other ADL -> ADL
}

# 5 类名（与 dataset.py LABEL_MAP 对齐）
CLASS5_NAMES = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]

# 推断红外/夜视的子数据集（待 probe 确认）
# UP-Fall 原数据集含红外 + 深度，CMDFall 多视角可能含红外
IR_DATASETS_HINT = {"up-fall", "upfall", "cmdfall", "mcfd"}


def adapt_omnifall(config: str = "of-sta-cs", out_dir: str = "./data/omnifall"):
    """从 HF datasets 拉取 OmniFall 配置，导出 annotations + 视频路径清单

    config: OmniFall 配置名，如 of-sta-cs / of-sta-to-all-cs / of-syn
    out_dir: 输出目录
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[FAIL] pip install datasets -q")
        return False

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anno_dir = out / "annotations"
    anno_dir.mkdir(exist_ok=True)

    print(f"[INFO] load_dataset('simplexsigil2/omnifall', '{config}') ...")
    ds = load_dataset("simplexsigil2/omnifall", config)

    summary = {"config": config, "splits": {}}

    for split, d in ds.items():
        cols = d.column_names
        print(f"\n=== {split}: n={len(d)} columns={cols} ===")

        # 推断字段名（OmniFall 可能用 path / video / video_path）
        path_key = None
        for cand in ["path", "video", "video_path", "file"]:
            if cand in cols:
                path_key = cand
                break
        if path_key is None:
            print(f"[FAIL] no video path field in {cols}, skip {split}")
            continue

        # label 字段
        label_key = None
        for cand in ["label", "labels", "activity", "class"]:
            if cand in cols:
                label_key = cand
                break
        if label_key is None:
            print(f"[FAIL] no label field in {cols}, skip {split}")
            continue

        # 子数据集字段（用于 IR 推断）
        src_key = None
        for cand in ["dataset", "source", "src", "db", "origin"]:
            if cand in cols:
                src_key = cand
                break

        # 视角/模态字段
        view_key = None
        for cand in ["view", "modality", "sensor", "camera"]:
            if cand in cols:
                view_key = cand
                break

        samples = []
        label16_dist = Counter()
        label5_dist = Counter()
        src_dist = Counter()
        ir_count = 0

        for s in d:
            vpath = s[path_key]
            if not vpath or not os.path.exists(vpath):
                # OmniFall path 可能是 HF cache 路径，video=True 才下载
                # 这里只存路径，读取时再决定
                pass
            label16 = int(s[label_key])
            label5 = OMNIFALL16_TO_5.get(label16, 0)

            src_name = s.get(src_key, "") if src_key else ""
            view_name = s.get(view_key, "") if view_key else ""

            # 红外推断：子数据集名 + 视角名含 ir/thermal/night
            is_ir = False
            src_lower = str(src_name).lower()
            view_lower = str(view_name).lower()
            if any(k in src_lower for k in IR_DATASETS_HINT):
                # UP-Fall 有红外视角，但不是所有视角都 IR，需要 view 字段细化
                # 保守：若 view 含 ir/thermal/night 才判为红外
                if any(k in view_lower for k in ["ir", "thermal", "night", "infrared"]):
                    is_ir = True
            if any(k in view_lower for k in ["ir", "thermal", "night", "infrared"]):
                is_ir = True
            if is_ir:
                ir_count += 1

            samples.append({
                "video_path": str(vpath),
                "label": label5,
                "original_label": label16,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", -1.0)),
                "scene": str(s.get("scene", "indoor")),
                "light": "infrared" if is_ir else str(s.get("light", "normal")),
                "source": str(src_name),
                "view": str(view_name),
                "is_ir": is_ir,
            })
            label16_dist[label16] += 1
            label5_dist[label5] += 1
            if src_name:
                src_dist[src_name] += 1

        # 写 annotations
        out_file = anno_dir / f"annotations_{split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[OK] {split}: {len(samples)} samples -> {out_file}")
        print(f"  16-class dist: {dict(label16_dist)}")
        print(f"  5-class dist:  {dict(label5_dist)}")
        print(f"  source dist:   {dict(src_dist)}")
        print(f"  IR samples:    {ir_count}")

        summary["splits"][split] = {
            "n": len(samples),
            "label16_dist": dict(label16_dist),
            "label5_dist": dict(label5_dist),
            "source_dist": dict(src_dist),
            "ir_count": ir_count,
            "annotation_file": str(out_file),
        }

    # 写 summary
    summary_path = anno_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] summary -> {summary_path}")
    print(f"\n[Done] Annotations under {anno_dir}")
    print(f"[Next] 视频本体需要 omnifall.load(config, video=True) 下载，"
          f"或对每个 video_path 用 predecode_videos.py 预解码")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="of-sta-cs",
                        help="OmniFall config，如 of-sta-cs / of-sta-to-all-cs / of-syn")
    parser.add_argument("--out", default="./data/omnifall")
    args = parser.parse_args()
    adapt_omnifall(config=args.config, out_dir=args.out)
