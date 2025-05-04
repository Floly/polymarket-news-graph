# Парсинг новостей
import requests
from bs4 import BeautifulSoup

def parse_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(responimport requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

def fetch_google_news_rss(query: str, cutoff_date: str, days_back: int = 7) -> list:
    """
    Fetches news from Google News RSS feed for a given query,
    filters by date (within [cutoff_date - days_back, cutoff_date]),
    and returns article URLs.

    Args:
        query (str): Search term for news (e.g., "Polymarket", "prediction market")
        cutoff_date (str): Target date in 'YYYY-MM-DD' format
        days_back (int): Number of days to look back (default: 7)

    Returns:
        list: List of URLs of relevant news articles
    """
    # Parse cutoff_date
    try:
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use 'YYYY-MM-DD'")

    # Calculate start date (cutoff_date - days_back)
    start_date = cutoff_dt - timedelta(days=days_back)

    # Build Google News RSS URL
    base_url = "https://news.google.com/rss/search"
    params = {
        'q': query,
        'hl': 'en-US',
        'gl': 'US'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Error fetching news: {e}")
        return []

    soup = BeautifulSoup(response.content, 'xml')

    # Extract all items (news articles)
    items = soup.find_all('item')

    urls = []

    for item in items:
        title_elem = item.find('title')
        link_elem = item.find('link')
        pub_date_elem = item.find('pubDate')

        if not all([title_elem, link_elem, pub_date_elem]):
            continue

        title = title_elem.text.strip()
        url = link_elem.text.strip()
        pub_date_str = pub_date_elem.text.strip()

        try:
            # Try parsing with timezone info
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            try:
                # Fallback without timezone
                pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S")
            except ValueError:
                print(f"⚠️ Could not parse date: {pub_date_str}")
                continue

        # Check if the article falls within the desired window
        if start_date <= pub_date <= cutoff_dt:
            urls.append(url)

    return urls


if __name__ == "__main__":
    # Example usage
    query = "Polymarket prediction market"
    cutoff_date = "2025-04-05"  # Date you're interested in
    days_back = 7  # Look back 7 days

    print(f"🔍 Searching for news about '{query}' between {cutoff_date} and {days_back} days before...")
    news_urls = fetch_google_news_rss(query, cutoff_date, days_back)

    if news_urls:
        print("\n✅ Found these articles:")
        for url in news_urls:
            print(url)
    else:
        print("❌ No recent articles found.")se.text, 'html.parser')
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