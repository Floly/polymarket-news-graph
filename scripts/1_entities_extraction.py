from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import json
import spacy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/question_entites_extraction.log'
)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)


# Configuration
NAME = "min_dt_2024-10-01"
RAW_DATA_PATH = f"../data/raw/results_{NAME}.json"
INTERIM_DATA_PATH = f"../data/interim/entities_{NAME}.json"

# Load data
try:
    events = json.load(open(RAW_DATA_PATH))
except Exception as e:
    logger.error(f"Failed to load raw data: {e}")
    exit(1)

# Load NLP models
try:
    spacy_nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "dslim/bert-base-NER", tokenizer_args={"do_basic_tokenize": False}
    )
    bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    bert_ner_pipeline = pipeline("ner", model=bert_model, tokenizer=tokenizer, aggregation_strategy="first")
except Exception as e:
    logger.error(f"Failed to load BERT NER model: {e}")
    exit(1)

# Dictionary to hold final results
event_entities = {}

# Load SpaCy for stopword detection
nlp = spacy.load("en_core_web_sm")

# Process each event
for event_data in tqdm(events):
    try:
        event = event_data.get("event")
        if not event:
            logger.warning("Missing 'event' key in event_data.")
            continue

        event_id = event.get("id")
        event_title = event.get("title")
        event_description = event.get("description")

        # Validate description
        if not isinstance(event_description, str):
            logger.warning(f"Invalid description for event ID {event_id}: {type(event_description)}")
            event_description = ""

        # Extract entities using BERT NER
        try:
            bert_entities = list({
                ent["word"].lower() 
                for ent in bert_ner_pipeline(event_description)
                if not nlp(ent["word"].lower())[0].is_stop
                }) if event_description else []
        except Exception as e:
            logger.error(f"BERT NER error for event {event_id}: {e}")
            bert_entities = []

        # Extract tags
        tags = event.get("tags", [])
        tag_labels = list({tag.get("label", "").lower() for tag in tags if tag.get("label")})
        tag_slugs = list({tag.get("slug", "") for tag in tags if tag.get("slug")})

        # Prepare event entity dictionary
        event_entities_dict = {
            "event_title": event_title,
            "event_ents_bert": bert_entities,
            "tag_labels": tag_labels,
            "tag_slugs": tag_slugs,
            "market_ents": {},
        }

        # Process market questions
        m_ents = {}
        markets = event_data.get("markets", [])
        for market_data in markets:
            market = market_data.get("market")
            if not market:
                logger.warning("Missing 'market' in market_data.")
                continue

            market_id = market.get("id")
            market_question = market.get("question")

            if not isinstance(market_question, str):
                logger.warning(f"Invalid market question for market ID {market_id}")
                market_question = ""

            # try:
            #     market_ents = list({
            #         ent["word"].lower() for ent in bert_ner_pipeline(market_question)
            #     }) if event_description else []
            # except Exception as e:
            #     logger.error(f"BERT NER error for event {event_id}: {e}")
            #     market_ents = []

            m_ents[market_id] = market_question

        event_entities_dict["market_ents"] = m_ents
        event_entities[event_id] = event_entities_dict

    except Exception as e:
        logger.error(f"Unexpected error processing event data: {e}")
        continue

# Save results to file
try:
    json.dump(event_entities, open(INTERIM_DATA_PATH, "w"), indent=2)
    logger.info(f"Results saved to {INTERIM_DATA_PATH}")
except Exception as e:
    logger.error(f"Failed to save results: {e}")