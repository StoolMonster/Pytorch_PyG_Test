# 神经网络通常以批处理方式进行训练。 PyG 通过创建稀疏块对角邻接矩阵（由 edge_index 定义）并在节点维度中连接特征和目标矩阵来实现小批量的并行化。
# 这种组合允许在一批中的示例上有不同数量的节点和边：

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch
    # >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    # >>> 32

