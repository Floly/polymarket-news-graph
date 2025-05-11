import os
import json
import logging
import numpy as np
from tqdm import tqdm
from itertools import combinations
import networkx as nx
from transformers import BertTokenizer, BertForTokenClassification, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/4_graph_building.log'
)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)

# Local imports
import sys
sys.path.insert(0, '..')
from utils.base import load_events  # Assuming this wraps `json.load`

# --- Paths ---
EVENTS_PATH = '../data/raw/results_min_dt_2025-03-01.json'
EMBEDDINGS_PATH = '../data/interim/sentence_embeddings/'
ARTICLES_PATH = '../data/interim/articles/'
OUTPUT_PATH = '../data/processed/graphs/'

SHARE_THRESHOLD = 0.35

# --- NER Setup ---
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    logger.info("NER pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load NER pipeline: {e}")
    raise


def extract_entities(title):
    """Extract unique named entities from a title using the NER pipeline."""
    try:
        ner_results = ner_pipeline(title)
        return set(entity['word'] for entity in ner_results if entity['entity'].startswith('B-'))
    except Exception as e:
        logger.warning(f"Error extracting entities from title: {e}")
        return set()


def stream_articles(file_path):
    """Stream articles from a JSON file one by one to reduce memory usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            for article in articles:
                yield article
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error in {file_path}: {e}")


def build_event_graph(event, articles_file_path):
    """Build a NetworkX graph based on article relationships for a single event."""
    ev_id = event['event']['id']
    logger.info(f"Building graph for event ID: {ev_id}")

    # Get embedded article IDs related to this event
    try:
        embedded_article_ids = {
            x.split('_')[2].split('.')[0] for x in os.listdir(EMBEDDINGS_PATH)
            if x.startswith(f"event_{ev_id}_")
        }
        logger.debug(f"Found {len(embedded_article_ids)} embedded article IDs for event {ev_id}")
    except Exception as e:
        logger.error(f"Error reading embeddings directory for event {ev_id}: {e}")
        return None

    valid_articles = []
    for article in stream_articles(articles_file_path):
        aid = article.get('id')
        if aid in embedded_article_ids:
            try:
                article['ner'] = extract_entities(article['title'])
                valid_articles.append(article)
            except Exception as e:
                logger.warning(f"Error processing article {aid}: {e}")

    if not valid_articles:
        logger.warning(f"No valid articles found for event {ev_id}")
        return None

    logger.info(f"Processing {len(valid_articles)} valid articles for event {ev_id}")

    # Build graph
    G = nx.Graph()

    # Add nodes
    for item in valid_articles:
        G.add_node(item['id'])

    # Add edges
    for a, b in combinations(valid_articles, 2):
        id_a, id_b = a['id'], b['id']
        same_date = a['date'] == b['date']
        same_query = a['query'] == b['query']
        same_publisher = a['publisher'] == b['publisher']

        common_entities = a['ner'].intersection(b['ner'])
        common_entities_share = len(common_entities) / max(len(a['ner']) or 1, len(b['ner']) or 1)

        if any([same_date, same_query, same_publisher, common_entities_share > SHARE_THRESHOLD]):
            G.add_edge(
                id_a,
                id_b,
                same_date=same_date,
                same_query=same_query,
                same_publisher=same_publisher,
                common_entities_share=common_entities_share
            )

    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges for event {ev_id}")
    return G


def main():
    try:
        events = load_events(EVENTS_PATH)
        logger.info(f"Loaded {len(events)} events from {EVENTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to load events from {EVENTS_PATH}: {e}")
        return

    existing_graphs = set(x.split('.')[0] for x in os.listdir(OUTPUT_PATH) if x.endswith('.graphml'))
    logger.debug(f"Existing graphs: {existing_graphs}")

    for i, event in enumerate(events):
        ev_id = event['event']['id']
        if ev_id in existing_graphs:
            logger.info(f"Skipping already processed event: {ev_id}")
            continue

        articles_file = f'{ARTICLES_PATH}{ev_id}_articles.json'

        if not os.path.exists(articles_file):
            logger.warning(f"Articles file does not exist: {articles_file}")
            continue

        logger.info(f"Starting graph construction for event {ev_id}")
        G = build_event_graph(event, articles_file)

        if G is not None:
            output_file = f"{OUTPUT_PATH}{ev_id}.graphml"
            nx.write_graphml(G, output_file)
            logger.info(f"Saved graph for event {ev_id} to {output_file}")
        else:
            logger.warning(f"Graph for event {ev_id} was not saved (None returned)")


if __name__ == '__main__':
    main()