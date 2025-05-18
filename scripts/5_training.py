import os
import json
import torch
import joblib
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dropout
from torch_geometric.utils.convert import from_networkx
from sklearn.utils.class_weight import compute_class_weight
import sys

# Project root and path setup
PREPROCESS = False
ROOT = '/Users/ivanesipov/Desktop/Учеба/МОВС_ВШЭ/Диплом/pm_news_graph/'
sys.path.insert(0, ROOT)

# from utils.models.gcn_v0 import GCNGraphClassifier
from utils.models.gcn_v1 import GCNGraphClassifier_v1
from utils.models.gcn_v2 import GCNGraphClassifier_v2
from utils.models.baseline import BaselineClassifier

CLF = GCNGraphClassifier_v2
# Data paths
GRAPHS_PATH = f'{ROOT}data/processed/graphs/'
EVENTS_PATH = f'{ROOT}data/raw/'
EMBEDDINGS_PATH = f'{ROOT}data/interim/sentence_embeddings/'

def get_similarity_vector(article_emb, question_emb, model):
    """Calculate similarity vector between article and question embeddings."""
    similarities_list = [
        model.similarity(s_emb, question_emb) 
        for s_emb in article_emb['arr_0']
    ]
    
    n = 10
    similarities = torch.cat(similarities_list)
    idx = similarities.argsort(dim=0, descending=True)[:n]
    
    article_similarity = torch.cat([
        torch.quantile(similarities, torch.tensor([i/10.0 for i in range(10)]), 
                      dim=0, keepdim=False).flatten(),
        torch.tensor([similarities.mean(), max(0, similarities[idx].std()), len(similarities > 0.5)]),
    ])
    
    return article_similarity

def clear_attrs(graph):
    """Remove specified attributes from graph."""
    keys = [
        'same_date',
        'common_entities_share',
        'same_query',
        'same_publisher'
    ]
    
    for key in keys:
        del graph[key]
    
    return graph

def generate_graph(market, G, model, event_id):
    """Generate graph with node embeddings and labels."""
    # Label
    pos = np.argmax(eval(market['outcomePrices']))
    y = np.where(eval(market['outcomes'])[pos] == 'No', 0, 1)
    
    # Question embedding
    market_question = market['question']
    question_emb = model.encode(market_question)
    
    for node_id in G.nodes:
        article_emb = np.load(f'{EMBEDDINGS_PATH}event_{event_id}_{node_id}.npz')
        node_vector = get_similarity_vector(article_emb, question_emb, model=model)
        G.nodes[node_id]['embedding'] = node_vector
    
    graph = from_networkx(G)
    graph.y = torch.tensor(y)
    
    return graph

def add_key(graph_list, key):
    """Add specified key to graphs if missing."""
    key = 'same_date'
    for g in graph_list:
        try:
            g[key]
        except KeyError:
            g[key] = torch.zeros(g.edge_index.size()[1])
    
    graph_list_new = [
        Data(
            x=g['embedding'],
            y=g['y'],
            edge_index=g['edge_index'],
            same_date=g[key]
        ) 
        for g in graph_list if g is not None
    ]
    return graph_list_new



def main():
    if PREPROCESS == True:
        # Initialize sentence transformer
        sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

        # Process all events
        all_events = [x for x in os.listdir(f'{ROOT}data/raw') if x.startswith('results')]
        for name in all_events:
            min_dt = name.split('_')[-1].split('.')[0]
            events = json.load(open(f'{ROOT}data/raw/{name}'))
            graphs = []
            
            for ev in tqdm(events):
                id = ev['event']['id']
                events_w_graph = [x.split('.')[0] for x in os.listdir(GRAPHS_PATH)]
                
                if id in events_w_graph:
                    G = nx.read_graphml(f'{GRAPHS_PATH}{id}.graphml')
                    for market in ev['event']['markets']:
                        try:
                            graphs.append(
                                generate_graph(market, G=G, model=sentence_transformer, event_id=id)
                            )
                        except Exception as e:
                            print(f"Error processing market: {e}")
                else:
                    continue
            
            graph_list = add_key(graphs, 'same_date')
            with open(f'{ROOT}data/processed/graph_data_4/graph_list_{min_dt}.pickle', 'wb') as f:
                joblib.dump(graph_list, f)

    # Load all graphs
    graph_list = []
    for gl in os.listdir(f'{ROOT}data/processed/graph_data_4/'):
        with open(f'{ROOT}data/processed/graph_data_4/{gl}', 'rb') as f:
            graph_list.extend(joblib.load(f))

    # Train-test split
    np.random.seed(0)
    N = len(graph_list)
    indices = np.arange(N).tolist()
    train_indices = np.random.choice(indices, size=int(N * 0.7), replace=False)
    test_indices = list(set(indices) - set(train_indices))

    train_data_loader = DataLoader(
        [graph_list[i] for i in train_indices],
        batch_size=16,
        shuffle=True
    )

    test_data_loader = DataLoader(
        [graph_list[i] for i in test_indices],
        batch_size=16
    )

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    input_dim = test_data_loader.dataset[0].x.size(1)
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLF(input_dim=input_dim, hidden_dim=128).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters())

    # Training loop
    k = 1
    loss_history = []

    targets = np.array([int(g.y) for g in graph_list])

    # Get unique class values and compute class frequencies
    classes = np.unique(targets)

    # Compute class weights using 'balanced' mode (inversely proportional to class frequencies)
    class_weights = compute_class_weight('balanced', classes=classes, y=targets)

    # Convert to PyTorch tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Add early stopping variables
    patience = 15         # Number of epochs to wait before stopping if no improvement
    min_val_loss = float('inf')
    counter = 0
    best_model = None    # Save best model weights

    for i, epoch in enumerate(range(40)):
        total_loss = 0
        model.train()
        
        for data in train_data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(torch.exp(out), data.y, weight=class_weights_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_history.append(loss.item())

        # Validation every k epochs
        if i % k == 0:
            print(f"\n==== Epoch {epoch} =====")
            print(f"Train Loss: {total_loss:.4f}")
            
            model.eval()
            correct = 0
            total_val_loss = 0
            
            with torch.no_grad():
                all_preds = []
                all_targets = []
                all_logits = []
                for data in test_data_loader:
                    data = data.to(device)
                    val_out = model(data)

                    val_loss = F.nll_loss(torch.exp(val_out), data.y, weight=class_weights_tensor)
                    total_val_loss += val_loss.item()

                    pred = val_out.max(1)[1].to(torch.float32)
                    correct += pred.eq(data.y).sum().item()

                    all_logits.append(val_out[:,1])
                    all_preds.append(pred)
                    all_targets.append(data.y)
            
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            all_logits = torch.cat(all_logits)

            val_accuracy = correct / len(test_data_loader.dataset)
            avg_val_loss = total_val_loss / len(test_data_loader)
            
            from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

            # Advanced metrics
            class_report = classification_report(all_targets, all_preds, digits=4)
            conf_matrix = confusion_matrix(all_targets, all_preds)
            
            precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
            roc_auc = roc_auc_score(all_targets, all_logits)

            print(f"Avg Prediction: {torch.mean(all_preds):.4f}")
            # print(f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(class_report)
            print("\nConfusion Matrix:")
            print(conf_matrix)

            print(f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                counter = 0
                # Save best model
                best_model = model.state_dict()  # Save current model state
            else:
                counter += 1
                if counter >= patience:
                    print("\nEarly stopping triggered.")
                    break

    class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'{ROOT}models/{class_name}_{timestamp}.pth'
    torch.save(best_model, model_path)
    
if __name__ == '__main__':
    main()