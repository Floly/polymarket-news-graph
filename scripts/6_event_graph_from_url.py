import json
from typing import List, Dict

import sys
sys.path.insert(0, '../')
from utils.parsers import PolymarketAPIClient, PriceHistoryFetcher
from utils.preprocessors import DataProcessor


# Temporary variables
START_MIN_DT = "2024-08-01"
START_MAX_DT = "2024-09-31"
RESULTS_PATH = f'../data/raw/results_min_dt_{START_MIN_DT}.json'
THRESHOLD = 0.8
SLUG = "fed-decision-in-june"


# Initialize clients
event_api = PolymarketAPIClient("https://gamma-api.polymarket.com/events")
price_api = PriceHistoryFetcher(PolymarketAPIClient("https://clob.polymarket.com/prices-history/"))
processor = DataProcessor()
# Initialize result storage
results: List[Dict] = []
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/data_gather.log'
)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(handler)

# Initialize clients
event_api = PolymarketAPIClient("https://gamma-api.polymarket.com/events")
price_api = PriceHistoryFetcher(PolymarketAPIClient("https://clob.polymarket.com/prices-history/"))

# Initialize result storage
results: List[Dict] = []

try:
    # Fetch events
    event_params = {
        "slug": SLUG
    }
    events = event_api.get_events(event_params)

    # Process each event
    for event_index, event in enumerate(events):  # Limit to first 1 event for testing
        logging.warning(f"\n=== Processing Event {event_index + 1} ===")

        # Process event data
        processed_event = processor.process_event(event)
        if not processed_event["markets"]:
            logging.warning("❌ No markets found in this event.")
            continue
        
        markets_list: List[Dict] = []
        # Process each market in the event
        for market_index, market in enumerate(processed_event["markets"]):
            logging.info(f"  --- Processing Market {market_index + 1} ---")

            # Validate required fields
            if not market.get("createdAt") or not market.get("closedTime"):
                logging.warning("  ❌ Missing 'createdAt' or 'closedTime' in market.")
                continue

            # Get max price index
            try:
                max_price_idx = processor.get_max_price_index(market["outcomePrices"])
            except Exception as e:
                logging.error(f"  ❌ Error getting max price index: {e}")
                continue

            # # Convert timestamps
            # try:
            #     created_at_ts = processor.date_converter.iso_or_yy_mm_dd_to_unix(market["createdAt"])
            #     closed_time_ts = processor.date_converter.iso_or_yy_mm_dd_to_unix(market["closedTime"])
            # except Exception as e:
            #     logging.error(f"  ❌ Error converting timestamps: {e}")
            #     continue

            # # Check time difference
            # TWO_DAYS_IN_SECONDS = 2 * 24 * 60 * 60
            # duration_seconds = closed_time_ts - created_at_ts
            # is_longer_than_two_days = duration_seconds > TWO_DAYS_IN_SECONDS

            # Fetch price history
            try:
                clob_token_id = processor.extract_clob_token_id(market["clobTokenIds"])
                price_history = price_api.fetch_prices_history(
                    clob_token_id,
                    # created_at_ts,
                    # closed_time_ts
                )
            except Exception as e:
                logging.error(f"  ❌ Error fetching price history: {e}")
                continue

            # Find last timestamp with price > threshold
            threshold = THRESHOLD  # You can change this value as needed
            last_timestamp = None
            history = price_history.get("history", [])

            for entry in reversed(history):
                try:
                    if float(entry["p"]) < threshold:
                        last_timestamp = entry["t"]
                        break
                except (KeyError, ValueError) as e:
                    logging.warning(f"  ⚠️ Error parsing price entry: {e}")

            # Save result
            markets_list.append({
                "market": market,
                "max_price_index": max_price_idx,
                # "created_at_ts": created_at_ts,
                # "closed_time_ts": closed_time_ts,
                # "duration_seconds": duration_seconds,
                # "is_longer_than_two_days": is_longer_than_two_days,
                "last_timestamp": last_timestamp,
                "price_threshold": threshold,
            })

            logging.info(f"  ✅ Finished processing market {market_index + 1}")
            
        results.append({
            "event": processed_event,
            "markets": markets_list
        })


    # Print final results
    logging.info("\n=== Final Results ===")
    for idx, res in enumerate(results):
        logging.info(f"\nResult {idx + 1}:")
        logging.info(f"  Event ID: {res['event'].get('title', 'N/A')}")
        logging.info("  Markets in Event: {}".format(len(res['markets'])))

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f)
        
    logging.warning(f'\nResults were saved to {RESULTS_PATH}')

except Exception as e:
    logging.critical(f"❌ Critical error in main(): {e}")
