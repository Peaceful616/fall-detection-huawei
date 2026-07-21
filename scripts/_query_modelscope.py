"""查询 ModelScope 跌倒数据集详情"""
import urllib.request
import json
import urllib.parse

names = ['Le2i', 'falldown_haikou', 'fall', 'fall_set', 'fall_detection_compress']
for name in names:
    url = f'https://modelscope.cn/api/v1/datasets/{name}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    try:
        r = urllib.request.urlopen(req, timeout=15)
        data = json.loads(r.read().decode('utf-8'))
        d = data.get('Data', {})
        print(f'\n=== {name} ===')
        print('  Name:', d.get('Name'))
        print('  ChineseName:', d.get('ChineseName'))
        print('  Description:', str(d.get('Description', ''))[:300])
        print('  License:', d.get('License'))
        print('  Categories:', d.get('Categories'))
        print('  Size:', d.get('Size'), d.get('DiskSize'))
        # 文件树
        try:
            tree_url = f'https://modelscope.cn/api/v1/datasets/{name}/repo/tree?Revision=master'
            req2 = urllib.request.Request(tree_url, headers={'User-Agent': 'Mozilla/5.0'})
            r2 = urllib.request.urlopen(req2, timeout=15)
            tree = json.loads(r2.read().decode('utf-8'))
            files = tree.get('Data', {}).get('Files', [])
            print(f'  Files ({len(files)}):')
            for f in files[:15]:
                print('   ', f.get('Path'), f.get('Size'))
        except Exception as e:
            print('  tree err:', e)
    except Exception as e:
        print(f'[{name}] ERR:', e)
