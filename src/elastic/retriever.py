import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.elastic.client import client
from src.elastic.models import SearchResponse, SearchParams, ChunkModel
from src.config import settings

class Retriever:
    """
    Модуль, отвечающий за фильтрацию и поиск релевантных чанков и их ранжирование.
    На вход принимает вопрос и параметры поиска, на выходе – релевантные вопросу чанки.

    Также отвечает за взаимодействие с движком поиска.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = client


    def search(self, query_embedding: List[float], search_params: Optional[SearchParams] = None) -> SearchResponse:
        """
        Векторный поиск чанков.

        Args:
            query_embedding: векторное представление запроса
            search_params: параметры поиска (фильтры, влияющие на выборку).

        Returns:
            SearchResult: результат поиска (чанки).
        """
        start_time = datetime.now()

        if search_params is None:
            search_params = SearchParams()

        try:
            search_body = self._build_vector_query(query_embedding, search_params)

            response = self.client.search(
                index=settings.index_name,
                body=search_body
            )

            chunks = self._parse_search_results(response)

            search_time = (datetime.now() - start_time).total_seconds()

            return SearchResponse(
                chunks=chunks,
                total_hits=response['hits']['total']['value'],
                search_time=search_time
            )

        except Exception as e:
            self.logger.error(f"Ошибка при поиске: {e}")

            return SearchResponse(
                chunks=[],
                total_hits=0,
                search_time=0.0
            )


    def delete_documents(self, ids: list[str]):
        """
        Удаляет все чанки документов, у которых document_id входит в список ids.
        """
        if not ids:
            raise ValueError("List of IDs is empty")

        query = {
            "query": {
                "terms": {
                    "document_id": ids
                }
            }
        }

        resp = self.client.delete_by_query(
            index=settings.index_name,
            body=query,
            refresh=True,
            conflicts="proceed",
        )

        return {
            "deleted": resp.get("deleted", 0),
            "total": resp.get("total", 0),
            "failures": resp.get("failures", []),
        }


    @staticmethod
    def _build_vector_query(query_embedding: List[float], search_params: SearchParams) -> Dict[str, Any]:
        query_filters = []

        date_range = {}

        if search_params.filter_date_from:
            date_range["gte"] = search_params.filter_date_from
        if search_params.filter_date_to:
            date_range["lte"] = search_params.filter_date_to

        if date_range:
            query_filters.append({"range": {"reg_date": date_range}})

        knn_query: Dict[str, Any] = {
            "field": "vector",
            "query_vector": query_embedding,
            "k": search_params.size or 10,
            "num_candidates": 100
        }

        if query_filters:
            knn_query["filter"] = {
                "bool": {
                    "filter": query_filters
                }
            }

        search_body = {
            "knn": knn_query,
            "size": search_params.size or 10
        }

        return search_body


    @staticmethod
    def _parse_search_results(response: Dict) -> List[ChunkModel]:
        chunks = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            source = hit['_source']

            chunk = ChunkModel(
                id=hit['_id'],
                document_id=source.get('document_id', ''),
                document_name=source.get('document_name', ''),
                file_name=source.get('file_name', ''),
                chunk_text=source.get('chunk_text', ''),
                total_chunks=source.get('total_chunks', ''),
                chunk_index=source.get('chunk_index', ''),
                reg_number=source.get('reg_number', ''),
                reg_date=source.get('reg_date', ''),
                score = score
            )
            chunks.append(chunk)

        return chunks