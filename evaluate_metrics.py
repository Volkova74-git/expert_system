import json
import numpy as np
import os   # <-- добавить
from elasticsearch import Elasticsearch
from gigachat import GigaChat
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "construction_standards"
TOP_K = 5
# ===============================

def load_giga():
    """Инициализация клиента GigaChat (как в app_streamlit.py)"""
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        api_base="https://gigachat.devices.sberbank.ru/api/v1",
        timeout=60
    )

def get_embedding(giga, text: str):
    """Получить эмбеддинг для текста через GigaChat"""
    response = giga.embeddings([text])
    return np.array(response.data[0].embedding, dtype=np.float32).tolist()

def search_es(query_text, giga, es, index_name, top_k=TOP_K, doc_filter=None):
    """
    Поиск релевантных чанков (возвращает список _id).
    Логика полностью совпадает с find_similar из app_streamlit.py.
    """
    query_vector = get_embedding(giga, query_text)
    base_query = {"match_all": {}}
    if doc_filter and doc_filter != "Все":
        base_query = {"term": {"doc_name.keyword": doc_filter}}
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": "(cosineSimilarity(params.query_vector, 'vector') + 1.0) / 2.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es.search(index=index_name, body=body)
    # Возвращаем список _id (строковых идентификаторов)
    return [hit['_id'] for hit in response['hits']['hits']]

# ---------- Метрики ----------
def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / len(retrieved_k)

def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def mrr(relevant, retrieved):
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevant, retrieved, k):
    rel = [1 if d in relevant else 0 for d in retrieved[:k]]
    if not any(rel):
        return 0.0
    dcg = sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
    ideal = sorted(rel, reverse=True)
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg

# ---------- Основная функция оценки ----------
def evaluate():
    # Загрузка тестовых запросов
    with open("test_queries.json", "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    # Подключение к Elasticsearch и GigaChat
    es = Elasticsearch(ES_HOST)
    giga = load_giga()

    results = []
    for q in test_queries:
        question = q["question"]
        relevant_ids = set(q["relevant_chunk_ids"])
        retrieved_ids = search_es(question, giga, es, INDEX_NAME, top_k=TOP_K)

        p1 = precision_at_k(relevant_ids, retrieved_ids, 1)
        r1 = recall_at_k(relevant_ids, retrieved_ids, 1)
        f1_1 = f1_score(p1, r1)
        p5 = precision_at_k(relevant_ids, retrieved_ids, 5)
        r5 = recall_at_k(relevant_ids, retrieved_ids, 5)
        f1_5 = f1_score(p5, r5)
        mrr_val = mrr(relevant_ids, retrieved_ids)
        ndcg_val = ndcg_at_k(relevant_ids, retrieved_ids, 5)

        results.append({
            "question": question,
            "relevant_ids": list(relevant_ids),
            "retrieved_ids": retrieved_ids,
            "precision@1": p1,
            "recall@1": r1,
            "f1@1": f1_1,
            "precision@5": p5,
            "recall@5": r5,
            "f1@5": f1_5,
            "mrr": mrr_val,
            "ndcg@5": ndcg_val
        })

        print(f"\nЗапрос: {question}")
        print(f"  Relevant IDs: {list(relevant_ids)[:5]}...")
        print(f"  Retrieved IDs: {retrieved_ids}")
        print(f"  Precision@1: {p1:.3f}, Recall@1: {r1:.3f}, F1@1: {f1_1:.3f}")
        print(f"  Precision@5: {p5:.3f}, Recall@5: {r5:.3f}, F1@5: {f1_5:.3f}")
        print(f"  MRR: {mrr_val:.3f}, NDCG@5: {ndcg_val:.3f}")

    # Агрегированные метрики
    metrics_keys = ["precision@1", "recall@1", "f1@1", "precision@5", "recall@5", "f1@5", "mrr", "ndcg@5"]
    aggregated = {}
    for key in metrics_keys:
        values = [r[key] for r in results]
        aggregated[f"mean_{key}"] = np.mean(values)
        aggregated[f"std_{key}"] = np.std(values)
        aggregated[f"min_{key}"] = np.min(values)
        aggregated[f"max_{key}"] = np.max(values)

    print("\n=== АГРЕГИРОВАННЫЕ МЕТРИКИ ===")
    for key, val in aggregated.items():
        print(f"{key}: {val:.4f}")

    # Сохранение в JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"individual": results, "aggregated": aggregated}, f, indent=2, ensure_ascii=False)
    print("\nРезультаты сохранены в evaluation_results.json")

if __name__ == "__main__":
    evaluate()