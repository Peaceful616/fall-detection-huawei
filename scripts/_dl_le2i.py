"""尝试下载 ModelScope Le2i 跌倒检测数据集

Le2i 是经典跌倒检测数据集，包含：
- 多个视频文件（.avi）
- 标注的跌倒时刻
- 室内场景（客厅、卧室、厨房、办公室）
"""
import os
import sys

# 配置
DATASET_ID = 'yangkailiang12/Le2i'
DEST_DIR = './data/le2i'

def try_download():
    from modelscope.hub.snapshot_download import snapshot_download
    print(f'Downloading {DATASET_ID} -> {DEST_DIR}')
    os.makedirs(DEST_DIR, exist_ok=True)
    try:
        local_path = snapshot_download(
            DATASET_ID,
            repo_type='dataset',
            cache_dir=DEST_DIR,
        )
        print(f'OK: {local_path}')
        return True
    except Exception as e:
        print(f'FAIL: {type(e).__name__}: {e}')
        return False

if __name__ == '__main__':
    ok = try_download()
    sys.exit(0 if ok else 1)
