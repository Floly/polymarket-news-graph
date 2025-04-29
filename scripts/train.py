# Парсинг новостей
import requests
from bs4 import BeautifulSoup

def parse_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [news.text for news in soup.select('.news-item')]

# # Генерация эмбеддингов
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(news_texts)

# # Построение графа
# import networkx as nx
# G = nx.Graph()
# for i, emb in enumerate(embeddings):
#     G.add_node(i, text=news_texts[i], embedding=emb)
# for i in range(len(embeddings)):
#     for j in range(i+1, len(embeddings)):
#         if cosine_similarity(embeddings[i], embeddings[j]) > 0.8:
#             G.add_edge(i, j)

# # Обучение GNN (PyTorch Geometric)
# from torch_geometric.nn import GCNConv
# import torch

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(768, 16)
#         self.conv2 = GCNConv(16, 2)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return torch.softmax(x, dim=1)

# # ... (подготовка данных, обучение, инференс)