"""OmniFall 数据集适配器

数据集来源：HuggingFace simplexsigil2/omnifall
- 16 类活动 taxonomy（core 0-9 + extended 10-15）
- 三大组件：OF-Staged (8 公开 lab 数据集) / OF-ItW (OOPS 真实意外) / OF-Syn (12000 合成视频)
- 70+ 配置，关键：
    of-sta-cs          8 staged 数据集 cross-subject 划分
    of-sta-cv          cross-view 划分
    of-sta-to-all-cs   训练 staged 测试 staged+itw+syn（cross-domain 泛化）
    of-syn             12000 合成视频，random 80/10/10

样本字段（probe 确认，与 STRUCTURE.md 一致）：
    {'path': 'adl/HopS6', 'label': 9, 'start': 0.0, 'end': 8.1,
     'subject': 6, 'cam': 1, 'dataset': 'caucafall'}

    path: 相对路径（dataset-specific root），逻辑 ID，不是绝对路径！
    label: 0-15 int（core 0-9，extended 10-15）
    subject: 受试者 id（-1 for ItW/Syn）
    cam: 摄像头视角 id（-1 for ItW/Syn）
    dataset: 子数据集名

10 类 core taxonomy（LABELS.md 确认）：
    0 walk, 1 fall, 2 fallen, 3 sit_down, 4 sitting,
    5 lie_down, 6 lying, 7 stand_up, 8 standing, 9 other

本适配器：
1. 用 datasets.load_dataset 拉取指定 config
2. 把 10/16 类 label 映射到本项目的 5 类
3. 导出 annotations_{train,val,test}.json 到 out_dir/annotations/
   字段与 kaggle_fall_adapter 对齐：
   {video_path(path 字段原值), label(int 0-4), original_label(int 0-15),
    start, end, subject, cam, dataset, is_ir(布尔)}
4. 视频文件本身需要另外下载：
   - OF-Syn: data_files/omnifall-synthetic_av1.tar（单一 tar，最简单）
   - OF-Staged: 需要各原始子数据集 + omnifall 包做 path→文件映射（复杂）
   - OF-ItW: prepare_oops_videos.py 提取 OOPS 视频

视频路径字段（video_path）在 annotations 里存的是 OmniFall 的 path 原值，
训练时需要通过 VideoFallDataset 的 video_path 解析逻辑找到真实文件。
对 OF-Syn，video_path = tar 内文件名（去掉 .mp4）。

用法：
    python data/omnifall_adapter.py --config of-sta-cs --out ./data/omnifall
    python data/omnifall_adapter.py --config of-syn --out ./data/omnifall_syn
"""
import argparse
import os
import json
from collections import Counter
from pathlib import Path


# 16 类 -> 5 类映射（LABELS.md 确认的 10 core + 6 extended）
# OmniFall 16:  0 walk, 1 fall, 2 fallen, 3 sit_down, 4 sitting,
#               5 lie_down, 6 lying, 7 stand_up, 8 standing, 9 other,
#               10 kneel_down, 11 kneeling, 12 squat_down, 13 squatting,
#               14 crawl, 15 jump
# 本项目 5 类: 0 ADL, 1 Fall, 2 Fall-like, 3 Lying, 4 Transition
OMNIFALL16_TO_5 = {
    0: 0,    # walk -> ADL
    1: 1,    # fall -> Fall
    2: 1,    # fallen -> Fall（跌倒后地面状态，题目要识别）
    3: 2,    # sit_down -> Fall-like（姿态下移，易误判）
    4: 0,    # sitting -> ADL
    5: 2,    # lie_down -> Fall-like（主动躺下，姿态下移易误判）
    6: 3,    # lying -> Lying
    7: 4,    # stand_up -> Transition
    8: 0,    # standing -> ADL
    9: 0,    # other -> ADL（兜底）
    10: 2,   # kneel_down -> Fall-like（屈膝下移）
    11: 0,   # kneeling -> ADL（静态跪姿，日常）
    12: 2,   # squat_down -> Fall-like（下蹲）
    13: 0,   # squatting -> ADL（静态蹲姿，日常）
    14: 0,   # crawl -> ADL（爬行，日常）
    15: 0,   # jump -> ADL（跳跃，日常）
}

# 5 类名（与 dataset.py LABEL_MAP 对齐）
CLASS5_NAMES = ["ADL", "Fall", "Fall-like", "Lying", "Transition"]

# OmniFall 16 类名（debug 用）
OMNIFALL16_NAMES = {
    0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
    5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other",
    10: "kneel_down", 11: "kneeling", 12: "squat_down", 13: "squatting",
    14: "crawl", 15: "jump",
}

# 推断红外/夜视的子数据集
# UP-Fall 原数据集含红外 + 淴度视角；但 OmniFall 的 path 不区分视角，
# cam 字段是视角 id，要靠 cam 值匹配红外视角（需要各子数据集原始文档确认）
# 保守策略：暂不标 IR，等拿到 UP-Fall 原始视角定义后再细化
IR_DATASETS_HINT = {"up_fall"}


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

        # OmniFall 字段固定为 path/label/start/end/subject/cam/dataset
        # (+ of-syn 扩展列，这里只取 core 7 列)
        required = ["path", "label", "start", "end", "dataset"]
        missing = [c for c in required if c not in cols]
        if missing:
            print(f"[FAIL] missing required columns {missing} in {cols}, skip {split}")
            continue

        samples = []
        label16_dist = Counter()
        label5_dist = Counter()
        src_dist = Counter()
        ir_count = 0

        for s in d:
            # path 是逻辑 ID（如 'adl/HopS6'），不是绝对路径
            # 真实视频文件需要另外解析：
            #   OF-Syn: tar 内文件名（path + '.mp4'）
            #   OF-Staged: 各子数据集原始视频 + omnifall 包做 path→文件映射
            logical_path = str(s["path"])
            label16 = int(s["label"])
            label5 = OMNIFALL16_TO_5.get(label16, 0)

            dataset_name = str(s.get("dataset", ""))
            subject = int(s.get("subject", -1)) if s.get("subject") is not None else -1
            cam = int(s.get("cam", -1)) if s.get("cam") is not None else -1

            # 红外推断：保守策略，UP-Fall 的 cam 值 5/6 对应红外+深度
            # （UP-Fall 原论文：cam1=RGB, cam2=RGB, cam3=depth, cam4=depth,
            #   cam5=infrared, cam6=infrared）
            # 但 OmniFall 是否保留红外视角未确认，先标记待后续验证
            is_ir = False
            if dataset_name.lower() in IR_DATASETS_HINT:
                # UP-Fall cam 5/6 可能是红外，需要实际验证
                if cam in (5, 6):
                    is_ir = True
            if is_ir:
                ir_count += 1

            samples.append({
                # video_path 存逻辑 ID，VideoFallDataset 读取时需解析为真实文件
                "video_path": logical_path,
                "label": label5,
                "original_label": label16,
                "original_label_name": OMNIFALL16_NAMES.get(label16, "unknown"),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", -1.0)),
                "subject": subject,
                "cam": cam,
                "dataset": dataset_name,
                "scene": "indoor",  # OmniFall 不区分场景，默认 indoor
                "light": "infrared" if is_ir else "normal",
                "is_ir": is_ir,
            })
            label16_dist[label16] += 1
            label5_dist[label5] += 1
            if dataset_name:
                src_dist[dataset_name] += 1

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
    print(f"[Next] 视频本体需另外下载：")
    print(f"  OF-Syn: 下载 data_files/omnifall-synthetic_av1.tar 后解压")
    print(f"  OF-Staged: 需各原始子数据集 + omnifall 包做 path->file 映射")
    print(f"  OF-ItW: 用 prepare_oops_videos.py 提取 OOPS 视频")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="of-sta-cs",
                        help="OmniFall config，如 of-sta-cs / of-sta-to-all-cs / of-syn")
    parser.add_argument("--out", default="./data/omnifall")
    args = parser.parse_args()
    adapt_omnifall(config=args.config, out_dir=args.out)
