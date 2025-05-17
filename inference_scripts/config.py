import os

# Root directories
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, 'data')
LOGS_ROOT = os.path.join(ROOT, 'logs')
MODELS_ROOT = os.path.join(ROOT, 'models')

# Inference/Event paths (to be set dynamically if needed)
EVENT_ID = None  # Set this at runtime
EVENT_PATH = lambda eid: os.path.join(DATA_ROOT, 'inference_data', str(eid))

# Model names
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
BERT_NER_MODEL = 'dslim/bert-base-NER'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Other constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"