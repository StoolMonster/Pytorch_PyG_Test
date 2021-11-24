# 下载Cora，一个标准基准数据集，用于半监督图节点分类

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
# >>> Cora()

len(dataset)
# >>> 1

dataset.num_classes
# >>> 7

dataset.num_node_features
# >>> 1433

print('a')