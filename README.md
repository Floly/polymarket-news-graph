# Сервис оценки ожиданий на рынке предсказаний с использованием новостного графа

Это репозиторий проекта, который использует модели глубинного графового обучения для предсказания исхода событий на рынке ставок [Polymarket](https://polymarket.com/)


### Мотивация

Машинное обучение на графах и NLP давно применяются на финансовых рынках для предсказания поведения финансовых инструментов на бирже. На этих рынках много крупных игроков, которые применяют эти технологии для торговли.

Площадка Polyamrket - рынок ставок, в котором игроки покупают исход события. События в отличие от финансовых инструментов гетерогенны, могут иметь произвольную дату окончания действия, имеют разнообразную природу возникновения. Сильным предиктором или даже итогом события может выступать новость. 

Новости могут быть смещены, субъективны и обладают большим количеством мета-информации. Например, дата выхода, процесс распространения, автор, площадка итд. Так как графы позволяют учитывать более сложные взаимосвязи, есть гипотеза, что использование графового машинного обучения позволит улучшить качество предсказания.

Задача проекта - собирать релевантные событию новости, объединять их в граф и обучать на нем модель предсказания исхода события.

### Структура репозитория

```
├── README.md         
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── utils              <- Utils
│   ├── __init__.py    <- Makes utils a Python module
|   |
|   ├── models
│   │   ├── __init__.py
│   │   ├── gcn_v0.py
|   |   └── baseline.py
│   │
│   │
│   ├── preprocessors.py 
│   ├── parsers.py 
│   └── base.py   
│
└── scripts             
    ├── __init__.py    <- Makes utils a Python module
    |
    ├── 0_market_data_gathering.py
    ├── 1_entities_extraction.py
    ├── 2_news_parsing.py
    ├── 3_vectorization.py
    ├── 4_graph_building.py
    └── 5_training.py
```