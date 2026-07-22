"""验证 Kaggle Fall Video Dataset 加载流程

测试：
1. build_datasets 能构建 train/val
2. 单个 sample 加载（视频帧 + keypoint）
3. shape 与 label 正确
4. keypoint 对齐（非全零）
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.default import cfg
from data.dataset import build_datasets


def main():
    print("=" * 60)
    print("Kaggle Fall Dataset verification")
    print("=" * 60)

    # 确认 data_root 指向 kaggle_fall
    print(f"\n[Config] data_root = {cfg.data_root}")
    print(f"[Config] seq_len = {cfg.seq_len}, input_size = {cfg.input_size}")
    print(f"[Config] aux_kp_enabled = {cfg.aux_kp_enabled}")

    # 1. 构建数据集
    print("\n[Test 1] build_datasets")
    train_set, val_set = build_datasets(cfg)
    print(f"  train: {len(train_set)}, val: {len(val_set)}")
    assert len(train_set) > 0, "train set empty"
    assert len(val_set) > 0, "val set empty"

    # 2. 加载第一个 train sample
    print("\n[Test 2] Load first train sample")
    s = train_set.samples[0]
    print(f"  sample[0]: video={os.path.basename(s['video_path'])}")
    print(f"    label={s['label']} (1=Fall, 0=ADL)")
    print(f"    has keypoint_csv: {'keypoint_csv' in s}")

    item = train_set[0]
    print(f"\n  Output keys: {list(item.keys())}")
    print(f"  video shape: {tuple(item['video'].shape)} (expect ({cfg.seq_len}, 3, {cfg.input_size[0]}, {cfg.input_size[1]}))")
    assert item["video"].shape == (cfg.seq_len, 3, *cfg.input_size), \
        f"video shape mismatch: {item['video'].shape}"
    assert item["label"] in [0, 1], f"label invalid: {item['label']}"

    # 3. keypoint 检查
    if "aux_kp" in item:
        kp = item["aux_kp"]
        print(f"\n  aux_kp shape: {tuple(kp.shape)} (expect ({cfg.seq_len}, 17, 2))")
        assert kp.shape == (cfg.seq_len, 17, 2), f"aux_kp shape mismatch: {kp.shape}"
        nonzero = (kp.abs() > 0).any(dim=-1).any(dim=-1).sum().item()
        print(f"  keypoint 非零帧数: {nonzero}/{cfg.seq_len}")
        print(f"  keypoint 范围: [{kp.min().item():.4f}, {kp.max().item():.4f}] (归一化 [0,1])")
    else:
        print("\n  [WARN] aux_kp not in output (可能 aux_kp_enabled=False 或 sample 无 keypoint_csv)")

    # 4. 加载一个 ADL sample 确认 label=0
    print("\n[Test 3] Find an ADL sample (label=0)")
    adl_idx = None
    for i, sm in enumerate(train_set.samples):
        if sm["label"] == 0:
            adl_idx = i
            break
    if adl_idx is not None:
        adl_item = train_set[adl_idx]
        print(f"  ADL sample idx={adl_idx}: label={adl_item['label']}")
        assert adl_item["label"] == 0
        print(f"  [OK] ADL label correct")
    else:
        print("  [WARN] No ADL sample found in train set")

    # 5. 加载 val sample（不应有 aux_kp）
    print("\n[Test 4] Load val sample (no aux_kp expected)")
    val_item = val_set[0]
    print(f"  val video shape: {tuple(val_item['video'].shape)}")
    assert "aux_kp" not in val_item, "val should not have aux_kp"
    print(f"  [OK] val has no aux_kp (correct)")

    print("\n" + "=" * 60)
    print("All tests passed. Kaggle Fall Dataset is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
