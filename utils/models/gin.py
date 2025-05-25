import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, BatchNorm
from torch_geometric.nn import Sequential as pygSequential


class GINClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.5, num_layers=3):
        super(GINClassifier, self).__init__()
        
        # Define GIN layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            
            conv = GINConv(mlp, train_eps=False)  # Allows learnable epsilon
            norm = BatchNorm(hidden_dim)

            self.convs.append(conv)
            self.norms.append(norm)

        # Final MLP classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.Dropout(dropout_p),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Global pooling (transform node embeddings to graph-level)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Final classification
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)