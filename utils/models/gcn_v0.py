import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GraphSAGE
import torch.nn.functional as F

class GCNGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(GCNGraphClassifier, self).__init__()

        # Define GNN layers
        self.sage = GraphSAGE(input_dim, hidden_dim, num_layers=3)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        # Non-linearities and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim, momentum=0.05)
        # self.batch_norm2 = nn.BatchNorm1d(hidden_dim*2)  # BatchNorm over hidden_dim

        # Final classifier
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        weights = data.same_date.float()

        # Scale weights: {0,1} -> {0.5, 1.0}
        weights = torch.where(weights == 1,
                              torch.tensor(1.0, device=weights.device),
                              torch.tensor(0.5, device=weights.device))

        # --- Layer 1: GraphSAGE ---
        x = self.sage(x, edge_index, edge_weight=weights)
        x = self.relu(x)

        # --- Layer 2: GCN ---
        x = self.conv1(x, edge_index, edge_weight=weights)
        x = self.dropout(x)
        x = self.relu(x)


        x = self.conv2(x, edge_index, edge_weight=weights)
        # x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_weight=weights)
        x = self.relu(x)

        # --- BatchNorm added here ---

        # --- Layer 3: GCN ---
        x = self.conv4(x, edge_index, edge_weight=weights)
        x = self.batch_norm1(x)
        x = self.relu(x)

        # --- Global pooling ---
        x = global_mean_pool(x, batch)

        # --- Final classifier ---
        x = self.lin(x)

        return F.log_softmax(x, dim=1)