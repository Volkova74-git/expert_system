import numpy as np
from elasticsearch import Elasticsearch
from gigachat import GigaChat
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding(giga, text):
    response = giga.embeddings([text])
    return np.array(response.data[0].embedding, dtype=np.float32).tolist()

def search(query, es, index_name, giga, top_k=5):
    query_vec = get_embedding(giga, query)
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }
    resp = es.search(index=index_name, body=body)
    results = []
    for hit in resp['hits']['hits']:
        results.append({
            "score": hit['_score'],
            "text": hit['_source']['chunk_text'],
            "doc_id": hit['_source'].get('doc_id', hit['_id'])
        })
    return results

if __name__ == "__main__":
    giga = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        #credentials="",
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro"
    )
    es = Elasticsearch("http://127.0.0.1:9200")
    results = search("отслоение кровли", es, "construction_standards", giga)
    for r in results:
        print(f"Score: {r['score']:.4f}\nText: {r['text'][:200]}...\n")