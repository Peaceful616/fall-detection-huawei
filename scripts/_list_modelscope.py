"""查询 ModelScope 跌倒数据集完整列表"""
from modelscope import HubApi

api = HubApi()
# 搜索 fall
for query in ['fall', '跌倒', 'fall detection', '摔倒']:
    print(f'\n===== Search: {query} =====')
    res = api.list_datasets(search=query, page_size=30, page_number=1)
    items = res.items if hasattr(res, 'items') else []
    print(f'Total count: {res.total_count}, page items: {len(items)}')
    for d in items:
        if not isinstance(d, dict):
            # RepoInfo 对象
            info = d
            print(f'  - {info.id} | downloads={info.downloads} | {info.description[:100] if info.description else ""}')
