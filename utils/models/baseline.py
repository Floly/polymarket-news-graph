import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class BaselineClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()

        # Final classifier
        self.lin1 = nn.Linear(input_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, batch = data.x, data.batch
        # --- Final classifier ---
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)