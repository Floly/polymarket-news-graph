import logging
from config import LOGS_ROOT, LOG_LEVEL, LOG_FORMAT

def setup_logger(event_id=None):
    log_file = f"{LOGS_ROOT}/{event_id or 'default'}.log"
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        filename=log_file
    )
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    return logging.getLogger(__name__)