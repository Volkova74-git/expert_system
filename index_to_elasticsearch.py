import pickle
import numpy as np
import sys
from elasticsearch import Elasticsearch, helpers

# ========== НАСТРОЙКИ ==========
INDEX_NAME = "construction_standards"
# Название норматива – можно задать при запуске: python index_to_elasticsearch.py "ГОСТ Р 21.101-2020"
if len(sys.argv) > 1:
    DOC_NAME = sys.argv[1]
else:
    DOC_NAME = "Неизвестный норматив"
# ===============================

# Загрузка текстов и эмбеддингов (из текущей папки)
with open("standards_texts.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("standards_index.faiss", "rb") as f:
    embeddings = np.load(f)

print(f"Загружено {len(chunks)} чанков, размерность {embeddings.shape[1]}")

# Подключение к Elasticsearch
es = Elasticsearch("http://127.0.0.1:9200")

# Создаём индекс, если его нет
if not es.indices.exists(index=INDEX_NAME):
    mapping = {
        "mappings": {
            "properties": {
                "chunk_text": {"type": "text", "analyzer": "russian"},
                "vector": {
                    "type": "dense_vector",
                    "dims": embeddings.shape[1],
                    "index": True,
                    "similarity": "cosine"
                },
                "doc_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "doc_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Индекс {INDEX_NAME} создан.")
else:
    print(f"Индекс {INDEX_NAME} уже существует, добавляем документы.")

# Подготовка bulk-запроса (добавление, а не перезапись)
actions = []
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    # Генерируем уникальный ID: можно использовать doc_name + chunk_index
    doc_id = f"{DOC_NAME}_{i}"
    action = {
        "_index": INDEX_NAME,
        "_id": doc_id,   # уникальный ID, чтобы не перезаписывать существующие
        "_source": {
            "chunk_text": chunk,
            "vector": emb.tolist(),
            "doc_id": doc_id,
            "chunk_index": i,
            "doc_name": DOC_NAME
        }
    }
    actions.append(action)

    if len(actions) >= 1000:
        helpers.bulk(es, actions)
        actions = []

if actions:
    helpers.bulk(es, actions)

print(f"✅ Добавлено {len(chunks)} документов в индекс {INDEX_NAME} (норматив: {DOC_NAME})")