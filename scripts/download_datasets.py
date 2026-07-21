"""数据集下载脚本：多源备选方案

OmniFall 国内下不下来，提供 3 种备选方案：
1. huggingface-cli + 镜像站
2. hf-mirror.com 镜像
3. git clone 镜像

Kaggle Fall Video Dataset 已下载后，本脚本做格式适配。
"""
import os
import sys
import subprocess
import json
from pathlib import Path


# ============ OmniFall 备用下载方案 ============

OMNIFALL_MIRRORS = [
    # 方案1：hf-mirror.com 镜像（国内最稳定）
    {
        "name": "hf-mirror",
        "cmd": "HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
               "simplexsigil2/omnifall --repo-type dataset "
               "--local-dir {dst}",
    },
    # 方案2：git clone 镜像
    {
        "name": "git-clone-mirror",
        "cmd": "git clone https://hf-mirror.com/datasets/simplexsigil2/omnifall {dst}",
    },
    # 方案3：直链 git clone（huggingface 直链）
    {
        "name": "git-clone-direct",
        "cmd": "git clone https://huggingface.co/datasets/simplexsigil2/omnifall {dst}",
    },
    # 方案4：huggingface_hub Python API（支持断点续传）
    {
        "name": "huggingface-hub-api",
        "cmd": "python -c \""
               "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; "
               "from huggingface_hub import snapshot_download; "
               "snapshot_download('simplexsigil2/omnifall', repo_type='dataset', "
               "local_dir='{dst}')\"",
    },
]


def try_download_omnifall(dst_dir: str) -> bool:
    """尝试多种方案下载 OmniFall"""
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    if os.path.exists(dst_dir) and os.listdir(dst_dir):
        print(f"[OK] OmniFall already exists at: {dst_dir}")
        return True

    print(f"\n[OmniFall] Trying {len(OMNIFALL_MIRRORS)} mirror options...")
    for i, m in enumerate(OMNIFALL_MIRRORS):
        print(f"\n[{i+1}/{len(OMNIFALL_MIRRORS)}] Trying: {m['name']}")
        cmd = m["cmd"].format(dst=dst_dir)
        print(f"  CMD: {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, timeout=600,
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  [OK] Download success: {m['name']}")
                return True
            else:
                print(f"  [FAIL] returncode={result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {m['name']} exceeded 10min")
        except Exception as e:
            print(f"  [ERROR] {e}")
    print(f"\n[FAIL] All mirrors failed for OmniFall")
    return False


# ============ UR Fall Detection Dataset（备选）============

def download_urfd(dst_dir: str) -> bool:
    """UR Fall Detection Dataset 直接下载（无需 HuggingFace）

    URL: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
    """
    if os.path.exists(dst_dir) and os.listdir(dst_dir):
        print(f"[OK] URFD already exists at: {dst_dir}")
        return True

    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    # URFD 提供直接下载链接
    urls = [
        "http://fenix.univ.rzeszow.pl/~mkepski/ds/urfall-cams.zip",
        "http://fenix.univ.rzeszow.pl/~mkepski/ds/urfall-adl.zip",
    ]
    print(f"\n[URFD] Downloading to: {dst_dir}")
    print("  Note: URFD 服务器较慢，建议用浏览器手动下载后放到目录")
    print("  Manual URL: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html")
    for url in urls:
        try:
            cmd = f"curl -L -o {dst_dir}/{os.path.basename(url)} {url}"
            print(f"  CMD: {cmd}")
            result = subprocess.run(cmd, shell=True, timeout=600)
            if result.returncode != 0:
                print(f"  [FAIL] {url}")
                return False
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False
    return True


# ============ Kaggle 数据集适配 ============

def adapt_kaggle_dataset(kaggle_dir: str, out_dir: str):
    """把 Kaggle Fall Video Dataset 适配为我们的标注格式

    Kaggle 数据集结构（典型）：
        kaggle_dir/
        ├── fall/
        │   ├── video1.mp4
        │   └── ...
        └── adl/  (或 non_fall / normal)
            ├── video1.mp4
            └── ...

    输出格式（适配我们的 dataset.py）：
        out_dir/
        ├── annotations.json
        └── videos/
            ├── fall_xxx.mp4
            └── adl_xxx.mp4
    """
    if not os.path.isdir(kaggle_dir):
        print(f"[SKIP] Kaggle dir not exists: {kaggle_dir}")
        return False

    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)

    # 自动识别子目录命名
    fall_dirs = ["fall", "falls", "Fall"]
    adl_dirs = ["adl", "ADL", "non_fall", "nonfall", "normal", "Normal"]

    annotations = []
    n_fall = n_adl = 0

    # 扫描 fall 类
    for d in fall_dirs:
        src = os.path.join(kaggle_dir, d)
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            if not f.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            dst_name = f"fall_{f}" if not f.startswith("fall") else f
            dst_path = os.path.join(out_dir, "videos", dst_name)
            if not os.path.exists(dst_path):
                # Windows 下用 copy，避免硬链接权限问题
                import shutil
                shutil.copy(os.path.join(src, f), dst_path)
            annotations.append({
                "video": dst_name,
                "label": "fall",
                "start": 0.0, "end": -1.0,
                "scene": "indoor", "light": "normal",
            })
            n_fall += 1

    # 扫描 adl 类
    for d in adl_dirs:
        src = os.path.join(kaggle_dir, d)
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            if not f.lower().endwith((".mp4", ".avi", ".mov")) if False else not f.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            dst_name = f"adl_{f}" if not f.startswith("adl") else f
            dst_path = os.path.join(out_dir, "videos", dst_name)
            if not os.path.exists(dst_path):
                import shutil
                shutil.copy(os.path.join(src, f), dst_path)
            annotations.append({
                "video": dst_name,
                "label": "adl",
                "start": 0.0, "end": -1.0,
                "scene": "indoor", "light": "normal",
            })
            n_adl += 1

    # 写 annotations.json
    anno_path = os.path.join(out_dir, "annotations.json")
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\n[Kaggle Adapt] OK")
    print(f"  Fall videos: {n_fall}")
    print(f"  ADL videos:  {n_adl}")
    print(f"  Total:       {n_fall + n_adl}")
    print(f"  Annotations: {anno_path}")
    print(f"  Videos dir:  {os.path.join(out_dir, 'videos')}")
    return n_fall + n_adl > 0


# ============ 主流程 ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=[
        "omnifall", "urfd", "adapt_kaggle", "all"
    ], default="all")
    parser.add_argument("--kaggle_dir", default="./data/kaggle_fall_raw",
                       help="Kaggle 解压后的原始目录")
    parser.add_argument("--out_dir", default="./data/omnifall",
                       help="适配后的输出目录（与 dataset.py 的 data_root 一致）")
    args = parser.parse_args()

    ROOT = Path(__file__).parent.parent
    data_root = ROOT / "data"
    omnifall_dir = data_root / "omnifall"
    urfd_dir = data_root / "urfd"

    if args.action in ("omnifall", "all"):
        try_download_omnifall(str(omnifall_dir))

    if args.action in ("urfd", "all"):
        download_urfd(str(urfd_dir))

    if args.action in ("adapt_kaggle", "all"):
        adapt_kaggle_dataset(args.kaggle_dir, args.out_dir)


if __name__ == "__main__":
    main()
