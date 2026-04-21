# src/elastic/retriever.py
class Retriever:
    def __init__(self, es_client, index_name, giga_client):
        self.es = es_client
        self.index_name = index_name
        self.giga = giga_client

    def search(self, query_text, top_k=5, **kwargs):
        # получаем эмбеддинг
        response = self.giga.embeddings([query_text])
        query_vector = response.data[0].embedding
        # knn search
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
        es_response = self.es.search(index=self.index_name, body=body)
        chunks = []
        for hit in es_response['hits']['hits']:
            chunks.append({
                "document_id": hit['_source'].get('doc_id', hit['_id']),
                "score": hit['_score'],
                "chunk_text": hit['_source']['chunk_text']
            })
        # Возвращаем объект, похожий на тот, что ожидает Evaluator
        from types import SimpleNamespace
        return SimpleNamespace(chunks=chunks)