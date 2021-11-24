# 下载Cora，一个标准基准数据集，用于半监督图节点分类
import os
os.environ["http_proxy"] = "http://127.0.0.1:51837"
os.environ["https_proxy"] = "http://127.0.0.1:51837"

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# >>> Cora()

len(dataset)
# >>> 1

dataset.num_classes
# >>> 7

dataset.num_node_features
# >>> 1433

# 这里，这个数据集仅包含一个单一的、无向引用图
data = dataset[0]
# >>> Data(edge_index=[2, 10556], test_mask=[2708],
#          train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

data.is_undirected()
# >>> True

data.train_mask.sum().item()
# >>> 140

data.val_mask.sum().item()
# >>> 500

data.test_mask.sum().item()
# >>> 1000

print('a')