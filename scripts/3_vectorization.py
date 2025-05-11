import logging
from tqdm import tqdm
import json
import re
import gc
import unicodedata
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/3_vectorization.log'
)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Constants
ARTICLES_PATH = '../data/interim/articles/'
NAME = 'results_min_dt_2024-10-01'
EVENTS_FILE_PATH = f'../data/raw/{NAME}.json'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/ ",
}



def load_events(file_path):
    """Load event data from JSON file."""
    try:
        logger.info(f"Loading events from {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading events from {file_path}: {e}")
        return []


def filter_url(url_info):
    """Filter URLs based on conditions."""
    url = url_info.get('real_url', '')
    if not url:
        return None

    try:
        conditions_met = (
            url.count('/') > 3 and
            len(re.findall(r'(author|topic|tag|category|youtube)', url)) < 1
        )
        return url_info if conditions_met else None
    except Exception as e:
        logger.warning(f"Error filtering URL {url}: {e}")
        return None


def normalize_unicode_text(text):
    """Normalize Unicode characters in text."""
    try:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    except Exception as e:
        logger.warning(f"Error normalizing Unicode text: {e}")
        return text


def clean_text(text):
    """Clean and preprocess raw HTML text content."""
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        text = normalize_unicode_text(text)

        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)                # Remove emails
        text = re.sub(r'\d+', '', text)                    # Remove digits
        text = text.lower()

        return text
    except Exception as e:
        logger.warning(f"Error cleaning text: {e}")
        return ""


def extract_sentences(text):
    """Extract non-empty sentences from cleaned text."""
    try:
        sentences = clean_text(text).split('.')
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    except Exception as e:
        logger.warning(f"Error extracting sentences: {e}")
        return []


def process_event_articles(event, articles_path):
    """Process all articles for a single event."""
    try:
        event_id = event['event']['id']
        logger.info(f"Processing event ID: {event_id}")

        file_path = f'{articles_path}{event_id}_articles.json'
        with open(file_path, 'r') as f:
            urls = json.load(f)

        filtered_urls = list(filter(None, [filter_url(url) for url in urls]))
        for url_info in filtered_urls:

            url = url_info.get('real_url', 'Unknown URL')
            article_id = url_info.get('id', 'Unknown ID')

            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url} (Status {response.status_code})")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            sentences = extract_sentences(soup.text)
            
            del soup, response
            gc.collect()

            if not sentences:
                logger.warning(f"No valid sentences extracted from {url}")
                continue

            embeddings = model.encode(sentences)
            output_path = f'../data/interim/sentence_embeddings/event_{event_id}_{article_id}.npz'
            logger.info(f"Saving embeddings to {output_path}")

            with open(output_path, 'wb') as out_file:
                np.savez_compressed(out_file, embeddings)
            
            del sentences, embeddings
            gc.collect()

    except Exception as e:
        logger.error(f"Error processing event articles for ID {event_id}: {e}", exc_info=True)


def main():
    events = load_events(EVENTS_FILE_PATH)

    if not events:
        logger.error("No events loaded. Exiting.")
        return

    for idx, event in enumerate(tqdm(events, desc="Processing Events")):
        process_event_articles(event, ARTICLES_PATH)

        # Periodically trigger GC every N events
        if idx % 10 == 0:
            gc.collect()

    # Final cleanup
    del events
    gc.collect()
    
if __name__ == '__main__':
    main()