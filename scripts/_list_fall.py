"""查询 ModelScope liyaoling/fall 数据集文件结构"""
import urllib.request
import json
from collections import Counter

for ds_id in ['liyaoling/fall', 'yangkailiang12/fall_up', 'yangkailiang12/fall_detection_compress']:
    print(f'\n===== {ds_id} =====')
    url = f'https://modelscope.cn/api/v1/datasets/{ds_id}/repo/tree?Revision=master&Recursive=true'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    try:
        r = urllib.request.urlopen(req, timeout=15)
        data = json.loads(r.read().decode('utf-8'))
        files = data.get('Data', {}).get('Files', [])
        print(f'Total files: {len(files)}')
        prefixes = Counter()
        for f in files:
            path = f.get('Path', '')
            prefix = path.split('/')[0] if '/' in path else path
            prefixes[prefix] += 1
        print('By prefix:')
        for p, n in prefixes.most_common():
            print(f'  {p}: {n} files')
        print('Large files (>1MB):')
        for f in files:
            size = f.get('Size', 0)
            if size and size > 1024 * 1024:
                print(f'  {f.get("Path")}: {size/1024/1024:.1f}MB')
        # 总大小
        total = sum(f.get('Size', 0) or 0 for f in files)
        print(f'Total size: {total/1024/1024:.1f}MB')
    except Exception as e:
        print(f'ERR: {e}')
