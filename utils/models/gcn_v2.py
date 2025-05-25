import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm, EdgePooling

class GCNGraphClassifier_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.05, output_dim=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim * 2))
        for _ in range(3):
            self.convs.append(GCNConv(hidden_dim * 2, hidden_dim * 2))

        self.bns = nn.ModuleList([
            BatchNorm(hidden_dim * 2, momentum=0.1) for _ in range(len(self.convs))
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # Combine mean and max pool
        self.lin = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        weights = data.same_date.float()
        weights = torch.where(weights == 1,
                              torch.tensor(1.0, device=weights.device),
                              torch.tensor(0.7, device=weights.device))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=weights)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.dropout(x) 

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)