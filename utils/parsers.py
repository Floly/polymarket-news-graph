import re
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