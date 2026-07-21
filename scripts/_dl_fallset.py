"""尝试下载 ModelScope liyaoling/fall_set 摔倒数据集"""
import os

DATASETS = [
    'liyaoling/fall_set',
    'liyaoling/fall',
    'yangkailiang12/fall_detection_compress',
    'yangkailiang12/fall_up',
]

def try_download(dataset_id, dest_dir):
    from modelscope.hub.snapshot_download import snapshot_download
    os.makedirs(dest_dir, exist_ok=True)
    try:
        local_path = snapshot_download(
            dataset_id,
            repo_type='dataset',
            cache_dir=dest_dir,
        )
        return local_path
    except Exception as e:
        return f'ERR: {e}'

if __name__ == '__main__':
    for ds in DATASETS:
        print(f'\n=== Trying {ds} ===')
        result = try_download(ds, f'./data/{ds.split("/")[-1]}')
        print(result if isinstance(result, str) else f'OK: {result}')
