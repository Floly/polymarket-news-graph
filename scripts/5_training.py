import os
import json
import torch
import joblib
import wandb
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import sys

# Project root and path setup
PREPROCESS = True
ROOT = '/Users/ivanesipov/Desktop/Учеба/МОВС_ВШЭ/Диплом/pm_news_graph/'
run = wandb.init(
    entity='ivan-flops-HSE',
    project="graph", 
    name='GCN-bn-momentum-02',    
    config={
    "batch_size": 16,
    "epochs": 60,
    "learning_rate": 0.02,
    "weight_decay": 0.01,
    "momentum": 0.01,
    "betas": (0.9, 0.999),
    "early_stop_patience": 15,
    "dropout_p": 0.05,
    "hidden_dim": 64,
})
config = wandb.config

sys.path.insert(0, ROOT)

# from utils.models.gcn_v0 import GCNGraphClassifier
from utils.models.gcn_v1 import GCNGraphClassifier_v1
from utils.models.gcn_v2 import GCNGraphClassifier_v2
from utils.models.baseline import BaselineClassifier
from utils.models.gin import GINClassifier

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
    graph.market_id = market['id']
    
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
            same_date=g[key],
            market_id=g['market_id'],
        ) 
        for g in graph_list if g is not None
    ]
    return graph_list_new

def train_test_split_loader(graph_list, test_size=0.3, batch_size=16):
    np.random.seed(42)
    N = len(graph_list)
    indices = np.arange(N).tolist()
    train_indices = np.random.choice(indices, size=int(N * (1 - test_size)), replace=False)
    test_indices = list(set(indices) - set(train_indices))

    train_loader = DataLoader(
        [graph_list[i] for i in train_indices],
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        [graph_list[i] for i in test_indices],
        batch_size=batch_size
    )

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    input_dim = test_loader.dataset[0].x.size(1)
    return train_loader, test_loader, input_dim


# ----------------------
# Training Step
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    wandb.watch(model, log_freq=100)
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_logits = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pred = out.max(1)[1]

        all_preds.append(pred.cpu())
        all_targets.append(data.y.cpu())
        all_logits.append(out[:, 1].cpu())

        wandb.log({"train/step_loss": loss.item()})

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_logits = torch.cat(all_logits)

    avg_loss = total_loss / len(loader)

    try:
        precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
        roc_auc = roc_auc_score(all_targets, all_logits.detach().numpy())
    except ValueError:
        roc_auc = float('nan')

    # Log metrics
    wandb.log({
        "train/loss": avg_loss,
        "train/precision": precision,
        "train/recall": recall,
        "train/f1": f1,
        "train/roc_auc": roc_auc,
        "train/pred_mean": torch.mean(all_preds.float()),
    })



# ----------------------
# Validation Step
# ----------------------
def validate(model, loader, criterion, device, epoch):
    model.eval()
    correct = 0
    total_val_loss = 0
    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            val_loss = criterion(out, data.y)
            total_val_loss += val_loss.item()

            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()

            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())
            all_logits.append(out[:, 1].cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_logits = torch.cat(all_logits)

    accuracy = correct / len(loader.dataset)
    avg_loss = total_val_loss / len(loader)

    try:
        precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
        roc_auc = roc_auc_score(all_targets, all_logits)
    except ValueError:
        roc_auc = float('nan')

    # Log metrics
    wandb.log({
        "val/loss": avg_loss,
        "val/accuracy": accuracy,
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1,
        "val/roc_auc": roc_auc,
        "val/pred_mean": torch.mean(all_preds.float()),
        "epoch": epoch
    })

    # Print summary
    print(f"\nValidation Results Epoch {epoch}")
    print(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # Generate classification report and confusion matrix
    class_report = classification_report(all_targets, all_preds, digits=4, zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Log as text and table
    wandb.log({
        # "classification_report": wandb.Html(class_report),
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_targets.numpy(), preds=all_preds.numpy())
    })

    return avg_loss, accuracy


# ----------------------
# Helper Functions
# ----------------------
def compute_class_weights(loader):
    targets = np.array([int(data.y) for data in loader.dataset])
    classes = np.unique(targets)
    class_weights = compute_class_weight('balanced', classes=classes, y=targets)
    return torch.tensor(class_weights, dtype=torch.float)

def main_training_loop(model_class, train_loader, test_loader, input_dim, epochs=40, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout_p=config.dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    class_weights_tensor = compute_class_weights(train_loader)
    criterion = lambda out, y: F.cross_entropy(torch.exp(out), y, weight=class_weights_tensor)

    min_val_loss = float('inf')
    counter = 0
    best_model = None

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)

        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print("\nEarly stopping triggered.")
                break

    # Save best model
    class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'{ROOT}models/{class_name}_{timestamp}.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model, model_path)
    print(f"Best model saved at: {model_path}")
    
    model_artifact = wandb.Artifact(
        "trained-model", type="model",
        description="Trained NN model",
        metadata=dict(config))
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)
    wandb.finish()

def load_graphs(path=f'{ROOT}data/processed/graph_data_5/'):
    graph_list = []
    for gl in os.listdir(path):
        with open(os.path.join(path, gl), 'rb') as f:
            graph_list.extend(joblib.load(f))
    return graph_list



if __name__ == '__main__':
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
            with open(f'{ROOT}data/processed/graph_data_5/graph_list_{min_dt}.pickle', 'wb') as f:
                joblib.dump(graph_list, f)

    graph_list = load_graphs()
    train_loader, test_loader, input_dim = train_test_split_loader(graph_list, batch_size=config.batch_size)
    main_training_loop(CLF, train_loader, test_loader, input_dim, epochs=config.epochs, patience=config.early_stop_patience)