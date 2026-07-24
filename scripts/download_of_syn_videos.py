"""下载并解压 OF-Syn 视频包

OF-Syn = 12000 个合成跌倒视频，打包成单一 tar 文件：
    data_files/omnifall-synthetic_av1.tar  (AV1 编码 MP4)

下载到 ./data/omnifall_syn/videos/，解压后每个视频对应 annotations 里的
path 字段（path + '.mp4'）。

用法（在远程执行，已设 HF_ENDPOINT=https://hf-mirror.com）：
    python scripts/download_of_syn_videos.py
    # 或指定输出目录
    python scripts/download_of_syn_videos.py --out ./data/omnifall_syn/videos
"""
import argparse
import os
import tarfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("pip install huggingface_hub -q")


REPO = "simplexsigil2/omnifall"
TAR_FILENAME = "data_files/omnifall-synthetic_av1.tar"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./data/omnifall_syn/videos",
                        help="解压输出目录")
    parser.add_argument("--keep_tar", action="store_true",
                        help="保留 tar 文件（默认下载后解压完就删）")
    parser.add_argument("--tar_dir", default="./data/omnifall_syn",
                        help="tar 文件存放目录")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_dir = Path(args.tar_dir)
    tar_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载 tar
    # hf_hub_download 内部支持断点续传（用 .incomplete 文件），
    # 中断重启会自动续传下载部分。
    tar_path = tar_dir / "omnifall-synthetic_av1.tar"
    # hf 实际下载路径（filename 带 data_files/ 前缀，会建子目录）
    hf_actual_path = tar_dir / "data_files" / "omnifall-synthetic_av1.tar"

    if tar_path.exists():
        print(f"[INFO] tar already at short path: {tar_path}")
    elif hf_actual_path.exists():
        # 上次下载完但没 rename 就中断了，这次补 rename
        print(f"[INFO] tar at hf path, moving to short path ...")
        hf_actual_path.rename(tar_path)
    else:
        print(f"[INFO] Downloading {TAR_FILENAME} from {REPO} ...")
        print(f"  (this is ~9.1GB, hf_hub_download supports resume)")
        local = hf_hub_download(
            repo_id=REPO,
            filename=TAR_FILENAME,
            repo_type="dataset",
            local_dir=str(tar_dir),
        )
        actual = Path(local)
        if actual != tar_path:
            actual.rename(tar_path)
        print(f"[OK] tar -> {tar_path}")

    # 2. 解压（支持断点续传：跳过已完整解压的文件）
    # 用临时文件 + rename 保证原子性，避免半成品文件被误判为已完成
    print(f"[INFO] Extracting to {out} ...")
    expected_mp4 = 12000  # OF-Syn 标称 12000 视频
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        mp4_members = [m for m in members
                       if m.name.endswith(".mp4") and m.isfile()]
        print(f"  total mp4 in tar: {len(mp4_members)} (expected ~{expected_mp4})")
        extracted = 0
        skipped = 0
        for m in mp4_members:
            name = os.path.basename(m.name)
            if not name:
                continue
            dst = out / name
            # 跳过已完整解压的（size 匹配 tar 内大小才算完整）
            if dst.exists() and dst.stat().st_size == m.size:
                skipped += 1
                continue
            # 写临时文件，完成后原子 rename，避免半成品
            tmp = dst.with_suffix(".mp4.tmp")
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                with open(tmp, "wb") as fp:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        fp.write(chunk)
                    # 确保写盘（必须在 with 块内，文件未关闭时）
                    fp.flush()
                    os.fsync(fp.fileno())
                tmp.rename(dst)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f"  extracted {extracted} new ({skipped} skipped) ...")
            except Exception as e:
                print(f"  [WARN] failed {name}: {e}")
                if tmp.exists():
                    tmp.unlink()
    n_videos = len(list(out.glob("*.mp4")))
    print(f"[OK] extracted {extracted} new, skipped {skipped} existing "
          f"(total in dir: {n_videos})")
    if n_videos < expected_mp4 * 0.95:
        print(f"[WARN] only {n_videos}/{expected_mp4} videos, "
              f"rerun to continue extraction")

    # 3. 可选删 tar（只在解压完整时才删）
    if not args.keep_tar:
        if n_videos >= expected_mp4 * 0.95:
            try:
                tar_path.unlink()
                print(f"[INFO] removed tar: {tar_path}")
            except Exception as e:
                print(f"[WARN] failed to remove tar: {e}")
        else:
            print(f"[INFO] keep tar (extraction incomplete, rerun to resume)")

    print(f"\n[Done] Videos under {out}")
    print(f"[Next] 跑 omnifall_adapter.py --config of-syn 生成 annotations，")
    print(f"       然后用 predecode_videos.py 预解码（需改路径解析逻辑）")


if __name__ == "__main__":
    main()
