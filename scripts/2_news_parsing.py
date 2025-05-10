import logging
from datetime import datetime
import itertools
import random
import asyncio
import json
import sys
import os

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/2_news_parsing.log'
)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)
# --------------------------

# Add parent directory to Python path for imports
sys.path.insert(0, '../')
from utils.parsers import fetch_google_news_rss, get_real_url_async
from utils.base import get_last_date

NAME = 'min_dt_2025-03-01'

def get_extracted_events(path='../data/interim/articles/'):
    urls = os.listdir(path)
    result = [u.split('_')[0] for u in urls]
    return result
    
def fetch_news_for_queries(queries, cutoff_date):
    """Fetch news articles for a list of queries and a cutoff date."""
    logger.info(f"Fetching news for {len(queries)} queries (cutoff: {cutoff_date})")
    news_data = {}
    for query in queries:
        try:
            res = fetch_google_news_rss(query, cutoff_date=cutoff_date)
            if res:
                news_data.update(res)
        except Exception as e:
            logger.error(f"Error fetching news for query '{query}': {e}")
    return news_data

async def url_extractor(event_id, articles, num_parallel_tasks=10):
    """
    Extract real URLs from articles in parallel using a specified number of tasks.
    
    Args:
        event_id (str): ID of the event.
        articles (list): List of article dicts.
        num_parallel_tasks (int): Number of parallel coroutines to use.
    """
    logger.info(f"Processing {len(articles)} articles for event ID: {event_id}")
    
    # Sample 100 articles if total exceeds 100
    if len(articles) > 100:
        logger.warning(f"Exceeded 100 articles. Sampling 100 out of {len(articles)}.")
        articles = random.sample(articles, 100)
        logger.info(f"Sampled 100 articles for event ID: {event_id}")

    # Create chunks based on number of parallel tasks
    chunk_size = max(1, len(articles) // num_parallel_tasks)
    chunks = [
        articles[i:i + chunk_size]
        for i in range(0, len(articles), chunk_size)
    ]

    logger.info(f"Created {len(chunks)} chunks for parallel processing")

    # Run all chunks in parallel using get_real_url_async
    tasks = [get_real_url_async(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    combined_articles = []
    for result in results:
        combined_articles.extend(result)

    output_path = f'../data/interim/articles/{event_id}_articles.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(combined_articles, f)
        logger.info(f"Saved processed articles to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save articles for event ID {event_id}: {e}")

async def main():
    """Main function to process events and fetch related news articles."""
    raw_events_path = f'../data/raw/results_{NAME}.json'
    entities_path = f'../data/interim/entities_{NAME}.json'

    try:
        with open(raw_events_path, 'r') as f:
            raw_events = json.load(f)
        logger.info(f"Loaded {len(raw_events)} events from {raw_events_path}")
    except Exception as e:
        logger.critical(f"Failed to load events from {raw_events_path}: {e}")
        return

    try:
        with open(entities_path, 'r') as f:
            event_ents = json.load(f)
        logger.info(f"Loaded entity data from {entities_path}")
    except Exception as e:
        logger.critical(f"Failed to load entities from {entities_path}: {e}")
        return

    for event in raw_events:
        event_id = event['event']['id']
        extracted_events =get_extracted_events()
        if event_id in extracted_events:
            continue
        else:
            logger.warning(f"Starting processing for event ID: {event_id}")
            markets = event['markets']

            timestamp = get_last_date(markets)
            last_date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

            tags = event_ents[event_id]['tag_labels']
            tag_combinations = list(itertools.combinations(tags, len(tags)))
            tag_queries = [' '.join(combo) for combo in tag_combinations]

            # Collect all queries from different sources
            queries = []
            queries.extend(event_ents[event_id]['event_ents_bert'])
            queries.extend(tag_queries)
            queries.extend([m['market'].get('question') for m in markets])

            news_data = fetch_news_for_queries(queries, last_date_str)

            articles = [
                {**article, 'query': source}
                for source, articles_list in news_data.items()
                for article in articles_list
            ]

            await url_extractor(event_id, articles, num_parallel_tasks=20)


if __name__ == '__main__':
    asyncio.run(main())