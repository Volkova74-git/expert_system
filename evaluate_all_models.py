import json
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys

ES_HOST = "http://127.0.0.1:9200"

MODELS = {
    "e5_small": {
        "index": "construction_standards_e5_small",
        "hf_name": "intfloat/multilingual-e5-small",
    },
    "e5_large": {
        "index": "construction_standards_e5_large",
        "hf_name": "intfloat/multilingual-e5-large",
    },
    "rosberta": {
        "index": "construction_standards_rosberta",
        "hf_name": "ai-forever/ru-en-RoSBERTa",
    }
}

def get_query_embedding(model, query_text, model_name):
    if "e5" in model_name.lower():
        query_text = f"query: {query_text}"
    return model.encode(query_text)

def search_es(es, index_name, query_vector, k=5):
    script_query = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    resp = es.search(index=index_name, body=script_query)
    return [hit['_id'] for hit in resp['hits']['hits']]

def evaluate_model(es, model_name, model_cfg, test_queries, k=5, debug=False):
    print(f"\n🔍 Оценка {model_name}")
    model = SentenceTransformer(model_cfg["hf_name"])
    results = []
    for q in tqdm(test_queries, desc="Запросы"):
        q_vec = get_query_embedding(model, q["question"], model_name)
        retrieved = search_es(es, model_cfg["index"], q_vec.tolist(), k)
        relevant = set(q["relevant_chunk_ids"])
        retrieved_set = set(retrieved)

        if debug:
            print(f"\nЗапрос: {q['question']}")
            print(f"Найденные ID: {retrieved}")
            print(f"Релевантные ID: {list(relevant)[:5]}{'...' if len(relevant)>5 else ''}")

        precision = len(relevant & retrieved_set) / k
        recall = len(relevant & retrieved_set) / len(relevant) if relevant else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mrr = 0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                mrr = 1 / (i + 1)
                break
        dcg = sum([1 / np.log2(i+2) for i, doc in enumerate(retrieved) if doc in relevant])
        idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(relevant)))])
        ndcg = dcg / idcg if idcg > 0 else 0

        results.append({
            "query_id": q["query_id"],
            "question": q["question"][:50],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
        })
    df = pd.DataFrame(results)
    numeric_cols = ['precision', 'recall', 'f1', 'mrr', 'ndcg']
    avg = df[numeric_cols].mean().to_dict()
    avg["model"] = model_name
    return avg, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_queries", default="test_queries.json", help="Путь к файлу с тестовыми запросами")
    parser.add_argument("--k", type=int, default=5, help="Top-K")
    parser.add_argument("--debug", action="store_true", help="Выводить отладочную информацию")
    args = parser.parse_args()

    try:
        with open(args.test_queries, encoding='utf-8') as f:
            test_queries = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {args.test_queries} не найден")
        sys.exit(1)

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("❌ Elasticsearch не доступен. Запустите docker-compose up -d")
        sys.exit(1)

    all_avg = []
    for name, cfg in MODELS.items():
        if not es.indices.exists(index=cfg["index"]):
            print(f"⚠️ Индекс {cfg['index']} не найден, пропускаем {name}")
            continue
        avg, _ = evaluate_model(es, name, cfg, test_queries, args.k, debug=args.debug)
        all_avg.append(avg)

    if not all_avg:
        print("Нет данных для сравнения. Убедитесь, что хотя бы один индекс существует и содержит документы.")
        sys.exit(0)

    summary = pd.DataFrame(all_avg).set_index("model")
    summary = summary[["precision", "recall", "f1", "mrr", "ndcg"]]
    summary.columns = ["Prec@5", "Rec@5", "F1@5", "MRR", "NDCG@5"]
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (ЧЕРЕЗ ELASTICSEARCH)")
    print("="*60)
    print(summary.round(4).to_string())

    summary.to_csv("model_comparison_es.csv")
    metrics = ["Prec@5", "Rec@5", "MRR", "NDCG@5"]
    fig, axes = plt.subplots(1, 4, figsize=(16,5))
    for i, m in enumerate(metrics):
        summary_sorted = summary.sort_values(m, ascending=True)
        axes[i].barh(summary_sorted.index, summary_sorted[m])
        axes[i].set_title(m)
        axes[i].set_xlim(0,1)
        for bar in axes[i].containers[0]:
            w = bar.get_width()
            axes[i].text(w-0.02, bar.get_y()+bar.get_height()/2, f"{w:.3f}", ha="right", va="center")
    plt.suptitle("Сравнение моделей на Elasticsearch")
    plt.tight_layout()
    plt.savefig("model_comparison_es.png")
    plt.show()

if __name__ == "__main__":
    main()