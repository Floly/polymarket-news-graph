import re
import os
import uuid
import requests
from typing import Dict
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from playwright.async_api import async_playwright


class PolymarketAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_events(self, params: Dict) -> Dict:
        """
        Fetches events from Polymarket API.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            JSON response
        """
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()


class PriceHistoryFetcher:
    def __init__(self, api_client: PolymarketAPIClient):
        self.api_client = api_client

    def fetch_prices_history(
        self,
        market_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int = 1440
    ) -> Dict:
        """
        Fetches price history for a specific market.
        
        Args:
            market_id: Market ID
            start_ts: Start timestamp
            end_ts: End timestamp
            fidelity: Time interval in minutes
        
        Returns:
            JSON response
        """
        url = "https://clob.polymarket.com/prices-history/"
        params = {
            "market": market_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json()


def fetch_google_news_rss(query: str, cutoff_date: str, days_back: int = 7) -> dict:
    """
    Fetches news from Google News RSS feed for a given query,
    filters by date (within [cutoff_date - days_back, cutoff_date]),
    and returns a structured dictionary with URL, source, publisher, date, and ID.

    Args:
        query (str): Search term for news (e.g., "Polymarket", "prediction market")
        cutoff_date (str): Target date in 'YYYY-MM-DD' format
        days_back (int): Number of days to look back (default: 7)

    Returns:
        dict: Dictionary with structure {query: [list_of_article_dicts]}
    """
    # Parse cutoff_date
    try:
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use 'YYYY-MM-DD'")

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
        return {}

    soup = BeautifulSoup(response.content, 'lxml-xml')

    # Extract all items (news articles)
    items = soup.find_all('item')

    rss_news = []

    for item in items:
        title_elem = item.find('title')
        link_elem = item.find('link')
        pub_date_elem = item.find('pubDate')
        source_elem = item.find('source')  # Try to get the actual publisher

        if not all([title_elem, link_elem, pub_date_elem]):
            continue

        title = title_elem.text.strip()
        url = link_elem.text.strip()

        # Extract source name from URL (fallback)
        try:
            source_domain = re.search(r'https?://([^/]+)', url).group(1)
        except:
            source_domain = "Unknown"

        # Try to get the actual publisher from <source> tag
        publisher = "Unknown"
        if title is not None:
            publisher = title.split('-')[-1].strip()

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
            # Generate a unique ID (UUID)
            article_id = str(uuid.uuid4())

            rss_news.append({
                'id': article_id,
                'url': url,
                'source': source_domain,
                'publisher': publisher,
                'date': pub_date.strftime("%Y-%m-%d"),
                'title': title
            })

    # Return result in the required format
    return {query: rss_news}


async def get_real_article_url_playwright(proxy_url: str) -> str:
    """
    Uses Playwright (async) to resolve the real article URL from a Google News proxy link.
    
    Args:
        proxy_url (str): The Google News view-through link.

    Returns:
        str: The resolved article URL or empty string on failure.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            print(f"🧭 Navigating to {proxy_url}")
            await page.goto(proxy_url, timeout=10000)

            # Wait for possible JavaScript rendering
            await page.wait_for_timeout(5000)

            # Try to click the main article link (heuristic-based selector)
            try:
                await page.click("main a", timeout=5000)
            except Exception as e:
                print("🖱️ Click failed:", e)

            # Get final URL after navigation
            final_url = page.url
            await browser.close()

            if "google.com" not in final_url:
                print(f"🔗 Resolved URL: {final_url}")
                return final_url
            else:
                print("⚠️ Still on Google domain:", final_url)
                return ""

    except Exception as e:
        print(f"❌ Error resolving via Playwright: {e}")
        return ""

    
async def get_real_url_async(articles):
    for article in articles:
        proxy_url = article['url']
        real_url = await get_real_article_url_playwright(proxy_url)
        article.update({'real_url': real_url})
        print("✅ Real Article URL:", real_url)
            
    return articles

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
