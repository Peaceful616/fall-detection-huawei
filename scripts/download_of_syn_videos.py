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
    tar_path = tar_dir / "omnifall-synthetic_av1.tar"
    if tar_path.exists():
        print(f"[INFO] tar already exists: {tar_path}")
    else:
        print(f"[INFO] Downloading {TAR_FILENAME} from {REPO} ...")
        print(f"  (this is ~9.1GB, may take a while)")
        local = hf_hub_download(
            repo_id=REPO,
            filename=TAR_FILENAME,
            repo_type="dataset",
            local_dir=str(tar_dir),
        )
        # hf_hub_download 会在 local_dir 下按相对路径建子目录
        # 实际文件在 tar_dir/data_files/omnifall-synthetic_av1.tar
        actual = Path(local)
        if actual != tar_path:
            # 移动到更短路径
            actual.rename(tar_path)
        print(f"[OK] tar -> {tar_path}")

    # 2. 解压
    print(f"[INFO] Extracting to {out} ...")
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        print(f"  total members in tar: {len(members)}")
        # 逐个解压，跳过已存在的
        extracted = 0
        for m in members:
            # tar 内文件名可能带前缀目录，取 basename
            name = os.path.basename(m.name)
            if not name or not name.endswith(".mp4"):
                continue
            dst = out / name
            if dst.exists() and dst.stat().st_size > 0:
                continue
            # 提取单个文件
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(dst, "wb") as fp:
                # 分块写
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    fp.write(chunk)
            extracted += 1
            if extracted % 1000 == 0:
                print(f"  extracted {extracted} videos ...")
        print(f"[OK] extracted {extracted} new videos (total in dir: {len(list(out.glob('*.mp4')))})")

    # 3. 可选删 tar
    if not args.keep_tar:
        try:
            tar_path.unlink()
            print(f"[INFO] removed tar: {tar_path}")
        except Exception as e:
            print(f"[WARN] failed to remove tar: {e}")

    print(f"\n[Done] Videos under {out}")
    print(f"[Next] 跑 omnifall_adapter.py --config of-syn 生成 annotations，")
    print(f"       然后用 predecode_videos.py 预解码（需改路径解析逻辑）")


if __name__ == "__main__":
    main()
