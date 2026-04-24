# index_embeddings_to_es.py
import json
import numpy as np
from elasticsearch import Elasticsearch, helpers
from pathlib import Path

ES_HOST = "http://127.0.0.1:9200"

# Где искать .npy файлы (несколько возможных папок)
SEARCH_DIRS = [
    Path("benchmark/embeddings"),
    Path("embeddings"),
    Path("data/processed"),
]

# Модели: имя для отображения, имя файла, желаемый индекс
MODELS = [
    ("e5_small", "chunk_embeddings_e5_small.npy", "construction_standards_e5_small"),
    ("e5_large", "chunk_embeddings_e5_large.npy", "construction_standards_e5_large"),
    ("rosberta", "chunk_embeddings_rosberta.npy", "construction_standards_rosberta"),
]

def find_file(filename):
    """Ищет файл в нескольких папках, возвращает первый найденный Path или None."""
    for search_dir in SEARCH_DIRS:
        candidate = search_dir / filename
        if candidate.exists():
            return candidate
    return None

def load_chunks_text():
    """Строит {chunk_id: text} из исходных JSON (как в генерации)."""
    STANDARDS_DIR = Path("standards/processed")
    chunks = {}
    for json_path in STANDARDS_DIR.glob("*.json"):
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        full_text = ""
        for page in doc.get("pages", []):
            full_text += page.get("page_content", "") + "\n\n"
        raw_chunks = full_text.split('\n\n')
        doc_name = json_path.stem
        for i, chunk in enumerate(raw_chunks):
            chunk = chunk.strip()
            if chunk:
                chunk_id = f"{doc_name}_{i}"
                chunks[chunk_id] = chunk
    print(f"Загружено текстов для {len(chunks)} чанков")
    return chunks

def create_index(es, index_name, dim):
    if es.indices.exists(index=index_name):
        print(f"Удаляем старый индекс {index_name}")
        es.indices.delete(index=index_name)
    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": dim, "similarity": "cosine"}
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
    print(f"Индекс {index_name} создан (dim={dim})")

def index_documents(es, index_name, embeddings_dict, text_map):
    actions = []
    for doc_id, vector in embeddings_dict.items():
        text = text_map.get(doc_id)
        if text is None:
            # Пропускаем – если ключ не найден, это может быть нормой для малого числа
            continue
        actions.append({
            "_index": index_name,
            "_id": doc_id,
            "_source": {"text": text, "embedding": vector.tolist() if hasattr(vector, 'tolist') else vector}
        })
    if not actions:
        print("Нет документов для индексации (нет совпадающих ID)")
        return 0
    success, errors = helpers.bulk(es, actions, stats_only=False, raise_on_error=False)
    if errors:
        print(f"Пример ошибки: {errors[0]}")
    print(f"Загружено {success} документов в {index_name}")
    return success

def main():
    # 1. Загружаем тексты
    text_map = load_chunks_text()
    if not text_map:
        print("Не удалось загрузить тексты")
        return

    # 2. Подключаемся к Elasticsearch
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("Elasticsearch не доступен. Запустите docker-compose up -d")
        return

    # 3. Обрабатываем каждую модель
    for model_name, filename, index_name in MODELS:
        file_path = find_file(filename)
        if not file_path:
            print(f" Файл {filename} не найден в {SEARCH_DIRS}, пропускаем {model_name}")
            continue

        print(f"\n--- Индексация {model_name} из {file_path} ---")
        embeddings = np.load(file_path, allow_pickle=True).item()
        print(f"Загружено {len(embeddings)} векторов")

        # Определяем размерность по первому вектору
        first_vec = next(iter(embeddings.values()))
        dim = len(first_vec)
        print(f"Автоопределённая размерность: {dim}")

        # Проверка совпадения ID с текстами
        missing = set(embeddings.keys()) - set(text_map.keys())
        if missing:
            print(f" {len(missing)} ID из эмбеддингов отсутствуют в текстах (пример: {list(missing)[:2]})")
        else:
            print("Все ID совпадают")

        create_index(es, index_name, dim)
        index_documents(es, index_name, embeddings, text_map)

    print("\nИндексация завершена")

if __name__ == "__main__":
    main()