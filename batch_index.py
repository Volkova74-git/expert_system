import os
import json
import shutil
import numpy as np
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from gigachat import GigaChat
from sentence_splitter import SentenceSplitter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  

# Настройка
STANDARDS_DIR = "standards"
PROCESSED_DIR = os.path.join(STANDARDS_DIR, "processed")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "construction_standards"   


def load_giga():
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = SentenceSplitter(language="ru")
    sentences = splitter.split(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = " ".join(current_chunk.split()[-overlap:]) if overlap else ""
            current_chunk = overlap_text + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [ch for ch in chunks if len(ch) >= 50]

def extract_text(data):
    if 'pages' in data and isinstance(data['pages'], list):
        return "\n".join([page.get('page_content', '') for page in data['pages']])
    elif 'text' in data:
        return data['text']
    elif 'content' in data:
        return data['content']
    else:
        raise ValueError(f"Неизвестная структура JSON: {list(data.keys())}")

def ensure_index(es, dims=1024):
    try:
        es.indices.get(index=INDEX_NAME)
        print(f"Индекс {INDEX_NAME} уже существует.")
    except NotFoundError:
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_text": {"type": "text", "analyzer": "russian"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims,
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
    except Exception as e:
        print(f"Ошибка при работе с индексом: {e}")
        raise

def index_documents():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    es = Elasticsearch(ES_HOST)
    # Проверяем подключение
    try:
        es.info()
        print("Подключение к Elasticsearch установлено.")
    except Exception as e:
        print(f"Не удалось подключиться к Elasticsearch: {e}")
        return

    ensure_index(es)
    giga = load_giga()

    json_files = [f for f in os.listdir(STANDARDS_DIR) if f.endswith('.json')]
    if not json_files:
        print("Нет JSON-файлов в папке standards/")
        return

    for json_file in json_files:
        file_path = os.path.join(STANDARDS_DIR, json_file)
        doc_name = os.path.splitext(json_file)[0]
        print(f"\n📄 Обработка: {doc_name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            full_text = extract_text(data)
        except ValueError as e:
            print(f"   Ошибка: {e}")
            continue

        chunks = split_text(full_text)
        print(f"   Получено {len(chunks)} чанков")

        all_embeddings = []
        for chunk in tqdm(chunks, desc=f"Эмбеддинги {doc_name}"):
            response = giga.embeddings([chunk])
            emb = np.array(response.data[0].embedding, dtype=np.float32).tolist()
            all_embeddings.append(emb)

        actions = []
        for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            action = {
                "_index": INDEX_NAME,
                "_id": f"{doc_name}_{i}",
                "_source": {
                    "chunk_text": chunk,
                    "vector": emb,
                    "doc_id": f"{doc_name}_{i}",
                    "chunk_index": i,
                    "doc_name": doc_name
                }
            }
            actions.append(action)
            if len(actions) >= 1000:
                helpers.bulk(es, actions)
                actions = []
        if actions:
            helpers.bulk(es, actions)

        # Перемещаем обработанный файл
        shutil.move(file_path, os.path.join(PROCESSED_DIR, json_file))
        print(f"   Добавлено {len(chunks)} документов, файл перемещён в {PROCESSED_DIR}")

    print("\n Все документы успешно проиндексированы!")

if __name__ == "__main__":
    index_documents()
