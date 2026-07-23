"""抓取 OmniFall 仓库的 README / LABELS.md / statistics.md

在远程执行（已设 HF_ENDPOINT=https://hf-mirror.com）：
    python scripts/fetch_omnifall_meta.py

把输出的 README / LABELS.md 内容贴回来，我据此确认 16 类 label 定义。
"""
import os
from pathlib import Path

OUT = Path("./data/omnifall_meta")
OUT.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("pip install huggingface_hub -q")

REPO = "simplexsigil2/omnifall"
FILES = ["README.md", "LABELS.md", "statistics.md", "STRUCTURE.md", "CONFIGS.md"]

for fn in FILES:
    try:
        local = hf_hub_download(repo_id=REPO, filename=fn,
                                repo_type="dataset", local_dir=str(OUT))
        print(f"\n[OK] {fn} -> {local}")
        print(f"\n===== {fn} (first 6000 chars) =====")
        try:
            with open(local, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read()[:6000])
        except Exception as e:
            print(f"(read fail: {e})")
    except Exception as e:
        print(f"\n[skip] {fn}: {e}")

print(f"\n[Done] Files saved under {OUT}/")
