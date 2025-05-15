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
import sys

# Project root and path setup
PREPROCESS = False
ROOT = '/Users/ivanesipov/Desktop/Учеба/МОВС_ВШЭ/Диплом/pm_news_graph/'
sys.path.insert(0, ROOT)

# from utils.models.gcn_v0 import GCNGraphClassifier
from utils.models.gcn_v1 import GCNGraphClassifier_v1
from utils.models.baseline import BaselineClassifier

CLF = GCNGraphClassifier_v1
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

def generate_graph(market, G, model):
    """Generate graph with node embeddings and labels."""
    # Label
    pos = np.argmax(eval(market['outcomePrices']))
    y = np.where(eval(market['outcomes'])[pos] == 'No', 0, 1)
    
    # Question embedding
    market_question = market['question']
    question_emb = model.encode(market_question)
    
    for node_id in G.nodes:
        article_emb = np.load(f'{EMBEDDINGS_PATH}event_{id}_{node_id}.npz')
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
                                generate_graph(market, G=G, model=sentence_transformer)
                            )
                        except Exception as e:
                            print(f"Error processing market: {e}")
                else:
                    continue
            
            graph_list = add_key(graphs, 'same_date')
            with open(f'{ROOT}data/processed/graph_data_2/graph_list_{min_dt}.pickle', 'wb') as f:
                joblib.dump(graph_list, f)

    # Load all graphs
    graph_list = []
    for gl in os.listdir(f'{ROOT}data/processed/graph_data_3/'):
        with open(f'{ROOT}data/processed/graph_data_3/{gl}', 'rb') as f:
            graph_list.extend(joblib.load(f))

    # Train-test split
    np.random.seed(0)
    N = len(graph_list)
    indices = np.arange(N).tolist()
    train_indices = np.random.choice(indices, size=int(N * 0.8), replace=False)
    test_indices = list(set(indices) - set(train_indices))

    train_data_loader = DataLoader(
        [graph_list[i] for i in train_indices],
        batch_size=16
    )

    test_data_loader = DataLoader(
        [graph_list[i] for i in test_indices],
        batch_size=16
    )

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLF(input_dim=12, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Training loop
    k = 1
    loss_history = []

    targets = np.array([int(g.y) for g in graph_list])
    class_0_weight, class_1_weight = (1 - targets.mean()).astype(np.float32), targets.mean().astype(np.float32)
    class_weight = torch.tensor([class_0_weight, class_1_weight])
    
    for i, epoch in enumerate(range(10)):
        total_loss = 0
        model.train()
        
        for data in train_data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y, weight=class_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_history.append(loss.mean())
        
        if i % k == 0:
            print(f"\n==== Epoch {epoch} =====")
            print(f"Train Loss: {total_loss:.4f}")
            
            model.eval()
            correct = 0
            total_val_loss = 0
            
            with torch.no_grad():
                for data in test_data_loader:
                    data = data.to(device)
                    val_out = model(data)
                    
                    val_loss = F.nll_loss(val_out, data.y)
                    total_val_loss += val_loss.item()
                    
                    pred = val_out.max(1)[1]
                    correct += pred.eq(data.y).sum().item()
            
            val_accuracy = correct / len(test_data_loader.dataset)
            print(f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {total_val_loss:.4f}")

    class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'{ROOT}models/{class_name}_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    
if __name__ == '__main__':
    main()