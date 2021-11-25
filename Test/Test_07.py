import os
os.environ["http_proxy"] = "http://127.0.0.1:51837"
os.environ["https_proxy"] = "http://127.0.0.1:51837"

from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

dataset[0]
# >>> Data(pos=[2518, 3], y=[2518])
