import json

with open('../data/interim/entities_min_dt_2025-03-01.json', 'r') as f:
    event_ents = json.load(f)
    
for