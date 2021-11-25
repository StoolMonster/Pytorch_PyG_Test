import os
os.environ["http_proxy"] = "http://127.0.0.1:51837"
os.environ["https_proxy"] = "http://127.0.0.1:51837"

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6))

dataset[0]
# >>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
