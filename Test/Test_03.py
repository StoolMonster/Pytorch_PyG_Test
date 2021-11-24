from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# >>> ENZYMES(600)

len(dataset)
# >>> 600

dataset.num_classes
# >>> 6

dataset.num_node_features
# >>> 3

data = dataset[0]
# >>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

data.is_undirected()
# >>> True

# 数据集包含37个节点，每个节点有3个特征。总共有168/2=84条无向边，该图仅被分配给一个class。此外，该数据对象仅有一个graph-level target

# 可以使用slices, long or bool tensors来划分数据集。比如创建90/10的训练/测试划分：

train_dataset = dataset[:540]
# >>> ENZYMES(540)

test_dataset = dataset[540:]
# >>> ENZYMES(60)