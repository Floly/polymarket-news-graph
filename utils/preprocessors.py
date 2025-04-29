import requests
from bs4 import BeautifulSoup
import json
from .base import DateConverter
from typing import Dict

def preprocess_webpage(input_data):
    # Extract the required fields from the input dictionary
    title = input_data.get('title', '')
    url = input_data.get('url', '')

    # Check if the URL is valid
    if not url:
        raise ValueError("URL is missing in the input data")

    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        html_content = response.text

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the main text content
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = ' '.join([element.get_text(strip=True) for element in text_elements])

        # Basic processing: Remove excessive whitespace
        processed_text = ' '.join(text.split())

        # Create the output JSON
        output_data = {title: processed_text}

        return output_data

    except requests.RequestException as e:
        print('error parsing {}'.format(input_data.get('title', '')))
        return ''


class DataProcessor:
    def __init__(self):
        self.date_converter = DateConverter()

    def process_event(self, event: Dict) -> Dict:
        """Extracts key fields from an event."""
        event_keys = ['id', 'title', 'description', 'startDate', 'endDate', 'tags', 'markets']
        return {key: event.get(key, None) for key in event_keys}

    def process_market(self, market: Dict) -> Dict:
        """Extracts key fields from a market."""
        market_keys = ['id', 'description', 'outcomes', 'outcomePrices', 'createdAt', 'closedTime', 'clobTokenIds']
        return {key: market.get(key, None) for key in market_keys}

    def get_max_price_index(self, outcome_prices: str) -> int:
        """
        Finds the index of the maximum price in outcomePrices.
        
        Args:
            outcome_prices: String representation of a list of prices
            
        Returns:
            Index of the maximum price
        """
        try:
            prices = json.loads(outcome_prices)
            return prices.index(max(prices))
        except (json.JSONDecodeError, ValueError):
            raise ValueError("Invalid outcomePrices format")

    def extract_clob_token_id(self, clob_token_ids: str) -> str:
        """
        Extracts the first CLOB token ID from a string.
        
        Args:
            clob_token_ids: String representation of a list of IDs
            
        Returns:
            First token ID as a string
        """
        try:
            return json.loads(clob_token_ids)[0]
        except (json.JSONDecodeError, IndexError):
            raise ValueError("Invalid clobTokenIds format")
