"""查询 ModelScope fall_set 数据集完整文件结构"""
import urllib.request
import json
from collections import Counter

url = 'https://modelscope.cn/api/v1/datasets/liyaoling/fall_set/repo/tree?Revision=master&Recursive=true'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
try:
    r = urllib.request.urlopen(req, timeout=15)
    data = json.loads(r.read().decode('utf-8'))
    files = data.get('Data', {}).get('Files', [])
    print(f'Total files in repo: {len(files)}')
    prefixes = Counter()
    for f in files:
        path = f.get('Path', '')
        prefix = path.split('/')[0] if '/' in path else path
        prefixes[prefix] += 1
    print('\nBy prefix:')
    for p, n in prefixes.most_common():
        print(f'  {p}: {n} files')

    print('\nLarge files (>1MB):')
    for f in files:
        size = f.get('Size', 0)
        if size and size > 1024 * 1024:
            print(f'  {f.get("Path")}: {size/1024/1024:.1f}MB')
except Exception as e:
    print('ERR:', e)
