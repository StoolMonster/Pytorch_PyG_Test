import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv

# todo dataset =

class myModel(torch.nn.Module):
    def __init__(self) -> None:
        super(myModel, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features,dataset.num_node_features)
        self.conv2 = GCNConv(dataset.num_node_features,dataset.num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
