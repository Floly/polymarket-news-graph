import requests
from typing import Dict


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
