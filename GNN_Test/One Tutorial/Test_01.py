# https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

# create a Data object
# 定义图中节点的信息
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
# 图中有4个节点， 每个节点有一个2维向量特征值，标签y表示节点分类

# 图的连接（边）情况。
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)
# 注意，边缘索引的顺序与创建的Data对象无关，因为这些信息仅用于计算邻接矩阵。
# 所以edge_index中的2个向量内的元素顺序打乱也可以，但是对应关系不能乱

# 将上述节点和边的信息结合，就可以生成一个Data object
data = Data(x=x, y=y, edge_index=edge_index)
print(data)


# 自定义数据集的例子
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list[data
            for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
