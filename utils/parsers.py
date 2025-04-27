import requests
from typing import List, Dict

class PolymarketParser:
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://gamma-api.polymarket.com"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def parse(self, limit: int=10, other_params: Dict={}) -> List:
        """
        Scrape events from Polymarket API and return a structured DataFrame.

        Args:
            limit (int): Number of events to retrieve.

        Returns:
            pd.DataFrame: DataFrame with columns [title, start_date, end_date, prices, description].
        """
        url = f"{self.base_url}/markets"
        params = {
            'limit': limit,            
        }
        params.update(other_params)
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()#.get('data', [])[:limit]

            print(f"Successfully scraped {len(data)} events into a DataFrame.")
            return data

        except requests.RequestException as e:
            print(f"[Error] Failed to scrape Polymarket events: {e}")
            return []

class NewsapiParser:
    def __init__(self, api_key: str):
        """
        Initialize the NewsAPIParser with your NewsAPI key.

        Args:
            api_key (str): NewsAPI authentication key.
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def parse(self, query: str, max_results: int = 10, language: str = "en", from_dt: str=None, to_dt: str= None) -> List[Dict]:
        """
        Search for news articles matching a query.

        Args:
            query (str): Keywords or phrase to search for.
            max_results (int): Maximum number of articles to retrieve.
            language (str): Language of the news articles (default: English).

        Returns:
            List[Dict]: List of news articles with key info.
        """
        endpoint = f"{self.base_url}/everything"
        params = {
            "q": query,
            "pageSize": max_results,
            "language": language,
            "from_param": from_dt,
            "to": to_dt,
            "sortBy": "relevancy"
        }

        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            results = []
            for article in articles:
                results.append({
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                    "description": article.get("description")
                })

            print(f"Found {len(results)} articles for query: '{query}'.")
            return results

        except requests.RequestException as e:
            print(f"[Error] Failed to fetch news articles: {e}")
            return []
