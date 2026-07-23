"""归档当前轮次训练产物

把三教师 + 学生 ckpt + 日志统一复制到 checkpoints/round_<tag>/，
并生成一份 manifest.json 记录 epoch、F1、各文件 sha256、时间戳。

用法（在远程执行）：
    python scripts/archive_round.py --tag kaggle_50ep --epoch 50 --f1 0.9976
    # 或自动从 student_best.pt 读取 metadata
    python scripts/archive_round.py --tag kaggle_50ep

产物：
    checkpoints/round_kaggle_50ep/
    ├── student_best.pt
    ├── teacher_slowfast_best.pt
    ├── teacher_video_swin_best.pt
    ├── teacher_mvit_best.pt
    ├── distill.log
    └── manifest.json
"""
import argparse
import os
import sys
import json
import hashlib
import shutil
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "logs")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="归档标签，如 kaggle_50ep")
    parser.add_argument("--epoch", type=int, default=None,
                        help="训练轮次（若不指定，尝试从 student_best.pt 读取）")
    parser.add_argument("--f1", type=float, default=None,
                        help="验证集 F1（若不指定，尝试从 student_best.pt 读取）")
    parser.add_argument("--logs", nargs="*", default=["distill.log"],
                        help="要归档的日志文件名（相对 logs/）")
    parser.add_argument("--note", default="",
                        help="备注，写入 manifest.json")
    args = parser.parse_args()

    out_dir = os.path.join(CKPT_DIR, f"round_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 复制学生 ckpt
    student_src = os.path.join(CKPT_DIR, "student_best.pt")
    if not os.path.exists(student_src):
        print(f"[FAIL] {student_src} not found")
        sys.exit(1)

    # 尝试从 ckpt 元数据读 epoch/f1
    if args.epoch is None or args.f1 is None:
        try:
            ckpt = torch_load(student_src)
            if args.epoch is None:
                args.epoch = ckpt.get("epoch", None)
            if args.f1 is None:
                args.f1 = ckpt.get("f1", None)
        except Exception as e:
            print(f"[WARN] read student ckpt metadata failed: {e}")

    student_dst = os.path.join(out_dir, "student_best.pt")
    shutil.copy2(student_src, student_dst)
    print(f"[OK] student -> {student_dst}")

    # 2. 复制三教师 ckpt
    teacher_names = ["slowfast", "video_swin", "mvit"]
    teacher_files = []
    for name in teacher_names:
        src = os.path.join(CKPT_DIR, f"teacher_{name}_best.pt")
        if not os.path.exists(src):
            print(f"[WARN] {src} not found, skip")
            continue
        dst = os.path.join(out_dir, f"teacher_{name}_best.pt")
        shutil.copy2(src, dst)
        teacher_files.append(f"teacher_{name}_best.pt")
        print(f"[OK] teacher_{name} -> {dst}")

    # 3. 复制日志
    log_files = []
    for log_name in args.logs:
        src = os.path.join(LOG_DIR, log_name)
        if not os.path.exists(src):
            print(f"[WARN] log {src} not found, skip")
            continue
        dst = os.path.join(out_dir, log_name)
        # 日志可能重名，加前缀
        dst = os.path.join(out_dir, f"{log_name}")
        shutil.copy2(src, dst)
        log_files.append(log_name)
        print(f"[OK] log {log_name} -> {dst}")

    # 4. 生成 manifest.json
    def torch_load(path):
        import torch
        return torch.load(path, map_location="cpu")

    manifest = {
        "tag": args.tag,
        "archived_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "epoch": args.epoch,
        "f1": args.f1,
        "note": args.note,
        "files": {},
    }

    # 计算所有文件 sha256 + 大小
    for fname in ["student_best.pt"] + teacher_files + log_files:
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            continue
        manifest["files"][fname] = {
            "size_bytes": os.path.getsize(fpath),
            "sha256": sha256(fpath),
        }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] manifest -> {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
