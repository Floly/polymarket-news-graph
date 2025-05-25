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
from sklearn.model_selection import KFold
import sys

# Project root and path setup
PREPROCESS = False
ROOT = '/Users/ivanesipov/Desktop/Учеба/МОВС_ВШЭ/Диплом/pm_news_graph/'
run = wandb.init(
    entity='ivan-flops-HSE',
    project="graph", 
    name='GCN-crossval',    
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
        "k_folds": 5
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
    pos = np.argmax(eval(market['outcomePrices']))
    y = np.where(eval(market['outcomes'])[pos] == 'No', 0, 1)
    
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

def create_kfold_loaders(graph_list, k_folds=5, batch_size=16):
    """Create data loaders for k-fold cross-validation."""
    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = np.arange(len(graph_list))
    
    fold_loaders = []
    for train_idx, test_idx in kf.split(indices):
        train_loader = DataLoader(
            [graph_list[i] for i in train_idx],
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            [graph_list[i] for i in test_idx],
            batch_size=batch_size
        )
        fold_loaders.append((train_loader, test_loader))
    
    input_dim = graph_list[0].x.size(1)
    print(f"Created {k_folds} folds with input dimension: {input_dim}")
    return fold_loaders, input_dim

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
        precision, recall, f1, roc_auc = float('nan'), float('nan'), float('nan'), float('nan')

    return avg_loss, precision, recall, f1, roc_auc

# ----------------------
# Validation Step
# ----------------------
def validate(model, loader, criterion, device, epoch, fold):
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
        precision, recall, f1, roc_auc = float('nan'), float('nan'), float('nan'), float('nan')

    # Log metrics for this fold
    wandb.log({
        f"fold_{fold}/val/loss": avg_loss,
        f"fold_{fold}/val/accuracy": accuracy,
        f"fold_{fold}/val/precision": precision,
        f"fold_{fold}/val/recall": recall,
        f"fold_{fold}/val/f1": f1,
        f"fold_{fold}/val/roc_auc": roc_auc,
        f"fold_{fold}/val/pred_mean": torch.mean(all_preds.float()),
        "epoch": epoch
    })

    # Print summary
    print(f"\nFold {fold} Validation Results Epoch {epoch}")
    print(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # Generate classification report and confusion matrix
    class_report = classification_report(all_targets, all_preds, digits=4, zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Log as text and table
    wandb.log({
        f"fold_{fold}/confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_targets.numpy(), preds=all_preds.numpy())
    })

    return avg_loss, accuracy, precision, recall, f1, roc_auc

# ----------------------
# Helper Functions
# ----------------------
def compute_class_weights(loader):
    targets = np.array([int(data.y) for data in loader.dataset])
    classes = np.unique(targets)
    class_weights = compute_class_weight('balanced', classes=classes, y=targets)
    return torch.tensor(class_weights, dtype=torch.float)

def main_training_loop(model_class, fold_loaders, input_dim, epochs=40, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_folds = len(fold_loaders)
    
    # Store metrics for each fold to compute std
    fold_metrics = {
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': []
    }
    
    for fold, (train_loader, test_loader) in enumerate(fold_loaders):
        print(f"\nStarting Fold {fold + 1}/{k_folds}")
        model = model_class(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout_p=config.dropout_p).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
        class_weights_tensor = compute_class_weights(train_loader).to(device)
        criterion = lambda out, y: F.cross_entropy(torch.exp(out), y, weight=class_weights_tensor)

        min_val_loss = float('inf')
        counter = 0
        best_model = None

        for epoch in range(epochs):
            train_loss, train_precision, train_recall, train_f1, train_roc_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            
            # Log training metrics
            wandb.log({
                f"fold_{fold}/train/loss": train_loss,
                f"fold_{fold}/train/precision": train_precision,
                f"fold_{fold}/train/recall": train_recall,
                f"fold_{fold}/train/f1": train_f1,
                f"fold_{fold}/train/roc_auc": train_roc_auc
            })

            val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc = validate(
                model, test_loader, criterion, device, epoch, fold)

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                counter = 0
                best_model = model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print(f"\nEarly stopping triggered in Fold {fold + 1}.")
                    break

        # Store metrics for this fold
        fold_metrics['val_loss'].append(min_val_loss)
        fold_metrics['val_accuracy'].append(val_acc)
        fold_metrics['val_precision'].append(val_precision)
        fold_metrics['val_recall'].append(val_recall)
        fold_metrics['val_f1'].append(val_f1)
        fold_metrics['val_roc_auc'].append(val_roc_auc)

        # Save best model for this fold
        class_name = model.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'{ROOT}models/{class_name}_fold{fold}_{timestamp}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model, model_path)
        print(f"Best model for fold {fold + 1} saved at: {model_path}")
        
        model_artifact = wandb.Artifact(
            f"trained-model-fold-{fold}", type="model",
            description=f"Trained NN model for fold {fold + 1}",
            metadata=dict(config))
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    # Compute and log standard deviations of metrics
    metrics_std = {
        'val_loss_std': np.std(fold_metrics['val_loss']) if fold_metrics['val_loss'] else float('nan'),
        'val_accuracy_std': np.std(fold_metrics['val_accuracy']) if fold_metrics['val_accuracy'] else float('nan'),
        'val_precision_std': np.std(fold_metrics['val_precision']) if fold_metrics['val_precision'] else float('nan'),
        'val_recall_std': np.std(fold_metrics['val_recall']) if fold_metrics['val_recall'] else float('nan'),
        'val_f1_std': np.std(fold_metrics['val_f1']) if fold_metrics['val_f1'] else float('nan'),
        'val_roc_auc_std': np.std(fold_metrics['val_roc_auc']) if fold_metrics['val_roc_auc'] else float('nan')
    }
    
    # Compute mean metrics
    metrics_mean = {
        'val_loss_mean': np.mean(fold_metrics['val_loss']) if fold_metrics['val_loss'] else float('nan'),
        'val_accuracy_mean': np.mean(fold_metrics['val_accuracy']) if fold_metrics['val_accuracy'] else float('nan'),
        'val_precision_mean': np.mean(fold_metrics['val_precision']) if fold_metrics['val_precision'] else float('nan'),
        'val_recall_mean': np.mean(fold_metrics['val_recall']) if fold_metrics['val_recall'] else float('nan'),
        'val_f1_mean': np.mean(fold_metrics['val_f1']) if fold_metrics['val_f1'] else float('nan'),
        'val_roc_auc_mean': np.mean(fold_metrics['val_roc_auc']) if fold_metrics['val_roc_auc'] else float('nan')
    }
    
    wandb.log(metrics_std)
    wandb.log(metrics_mean)
    print("\nCross-Validation Results:")
    print(f"Mean Validation Loss: {metrics_mean['val_loss_mean']:.4f} ± {metrics_std['val_loss_std']:.4f}")
    print(f"Mean Validation Accuracy: {metrics_mean['val_accuracy_mean']:.4f} ± {metrics_std['val_accuracy_std']:.4f}")
    print(f"Mean Validation Precision: {metrics_mean['val_precision_mean']:.4f} ± {metrics_std['val_precision_std']:.4f}")
    print(f"Mean Validation Recall: {metrics_mean['val_recall_mean']:.4f} ± {metrics_std['val_recall_std']:.4f}")
    print(f"Mean Validation F1: {metrics_mean['val_f1_mean']:.4f} ± {metrics_std['val_f1_std']:.4f}")
    print(f"Mean Validation ROC AUC: {metrics_mean['val_roc_auc_mean']:.4f} ± {metrics_std['val_roc_auc_std']:.4f}")
    
    wandb.finish()

def load_graphs(path=f'{ROOT}data/processed/graph_data_5/'):
    graph_list = []
    for gl in os.listdir(path):
        with open(os.path.join(path, gl), 'rb') as f:
            graph_list.extend(joblib.load(f))
    return graph_list

if __name__ == '__main__':
    # if PREPROCESS == True:
    #     sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    #     all_events = [x for x in os.listdir(f'{ROOT}data/raw') if x.startswith('results')]
    #     for name in all_events:
    #         min_dt = name.split('_')[-1].split('.')[0]
    #         events = json.load(open(f'{ROOT}data/raw/{name}'))
    #         graphs = []
            
    #         for ev in tqdm(events):
    #             id = ev['event']['id']
    #             events_w_graph = [x.split('.')[0] for x in os.listdir(GRAPHS_PATH)]
                
    #             if id in events_w_graph:
    #                 G = nx.read_graphml(f'{GRAPHS_PATH}{id}.graphml')
    #                 for market in ev['event']['markets']:
    #                     try:
    #                         graphs.append(
    #                             generate_graph(market, G=G, model=sentence_transformer, event_id=id)
    #                         )
    #                     except Exception as e:
    #                         print(f"Error processing market: {e}")
    #             else:
    #                 continue
            
    #         graph_list = add_key(graphs, 'same_date')
    #         with open(f'{ROOT}data/processed/graph_data_5/graph_list_{min_dt}.pickle', 'wb') as f:
    #             joblib.dump(graph_list, f)

    graph_list = load_graphs()
    fold_loaders, input_dim = create_kfold_loaders(graph_list, k_folds=config.k_folds, batch_size=config.batch_size)
    main_training_loop(CLF, fold_loaders, input_dim, epochs=config.epochs, patience=config.early_stop_patience)