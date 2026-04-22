from elasticsearch import Elasticsearch

es = Elasticsearch("http://127.0.0.1:9200")
index_name = "construction_standards"

def search_chunks(keyword):
    body = {
        "query": {"match": {"chunk_text": keyword}},
        "size": 50
    }
    resp = es.search(index=index_name, body=body)
    hits = resp['hits']['hits']
    if not hits:
        print("Ничего не найдено.")
        return
    for hit in hits:
        print(f"\n--- ID: {hit['_id']} | Документ: {hit['_source']['doc_name']} ---")
        print(hit['_source']['chunk_text'][:500] + "...")
        input("Нажмите Enter для следующего...")

while True:
    kw = input("Введите ключевое слово (или 'exit'): ")
    if kw == 'exit':
        break
    search_chunks(kw)