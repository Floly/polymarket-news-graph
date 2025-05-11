from datetime import datetime
from typing import Optional
from datetime import datetime
import json


def load_events(file_path):
    """Load event data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return []

def get_last_date(markets):
    """
    Returns oldest last timestamp from list of markets markets
    """
    min_tmstmp = datetime.now().timestamp()
    for m in markets:
        if m['last_timestamp'] is not None:
            if m['last_timestamp'] < min_tmstmp:
                min_tmstmp = m['last_timestamp']
    
    return min_tmstmp

class DateConverter:
    @staticmethod
    def iso_or_yy_mm_dd_to_unix(date_str: str, date_format: Optional[str] = None) -> int:
        """
        Converts a date string (ISO or yy mm dd) to Unix timestamp.
        
        Args:
            date_str: Date string (e.g., "2024-04-05" or "24 04 05")
            date_format: Optional format string (e.g., "%Y-%m-%d")
        
        Returns:
            Unix timestamp in seconds
        """
        if date_format is None:
            if 'T' in date_str or ' ' in date_str:
                # ISO format with time
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except ValueError:
                    raise ValueError("Invalid ISO format")
            elif len(date_str.split()) == 3:
                # yy mm dd format
                dt = datetime.strptime(date_str, "%y %m %d")
            else:
                raise ValueError("Unsupported date format")
        else:
            dt = datetime.strptime(date_str, date_format)

        return int(dt.timestamp())