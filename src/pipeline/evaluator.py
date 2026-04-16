import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

from src.pipeline.reranker import Reranker


class MetricType(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MRR = "mrr"
    NDCG = "ndcg"


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rerank_score: float
    content: str
    rank: int


@dataclass
class EvaluationResult:
    question: str
    expected_answer: str
    expected_doc_id: str
    search_results: List[SearchResult]
    metrics: Dict[str, float]
    search_time: float


class Evaluator:
    def __init__(self, retriever, k_values: List[int] = None):
        """
        Инициализация Evaluator.

        Args:
            retriever: для поиска в Elasticsearch
            k_values: значения k для метрик Precision@k, Recall@k и т.д.
        """
        self.retriever = retriever
        self.k_values = k_values or [1, 5]
        self.logger = logging.getLogger(__name__)
        self.reranker = Reranker()

    def evaluate_benchmark(self, benchmark: pd.DataFrame,
                           embedder: Any = None,
                           search_params: Optional[Any] = None) -> Dict[str, Any]:
        """
        Основной метод оценки бенчмарка.

        Args:
            benchmark: DataFrame с бенчмарком (question, answer, doc_name, doc_id)
            embedder: для получения эмбеддингов запросов
            search_params: параметры поиска

        Returns:
            Dict с результатами оценки
        """
        results = []
        total_metrics = {f"{metric}@{k}": [] for metric in ['precision', 'recall', 'f1']
                         for k in self.k_values}
        total_metrics.update({'mrr': [], 'ndcg': []})

        for _, row in benchmark.iterrows():
            try:
                evaluation_result = self.evaluate_single_query(
                    question=row['question'],
                    expected_answer=row['answer'],
                    expected_doc_id=row['doc_id'],
                    embedder=embedder,
                    search_params=search_params
                )
                results.append(evaluation_result)

                # Агрегируем метрики
                for metric_name, value in evaluation_result.metrics.items():
                    if metric_name in total_metrics:
                        total_metrics[metric_name].append(value)

            except Exception as e:
                self.logger.error(f"Ошибка при оценке вопроса '{row['question']}': {e}")
                continue

        aggregated_metrics = self._aggregate_metrics(total_metrics)

        return {
            'individual_results': results,
            'aggregated_metrics': aggregated_metrics,
            'total_queries': len(results),
            'successful_queries': len([r for r in results if r.search_results])
        }

    def evaluate_single_query(self,
                              question: str,
                              expected_answer: str,
                              expected_doc_id: str,
                              embedder: Any = None,
                              search_params: Optional[Any] = None) -> EvaluationResult:
        """
        Оценка для одного запроса.
        """
        start_time = time.time()

        query_embedding = None
        if embedder:
            query_embedding = embedder.embed_query(question)

        search_response = self.retriever.search(
            query_embedding=query_embedding,
            search_params=search_params
        )

        search_response = self.reranker.rerank(question, search_response)

        search_time = time.time() - start_time

        search_results = []
        for i, chunk in enumerate(search_response.chunks):
            search_results.append(SearchResult(
                doc_id=getattr(chunk, 'document_id', 'unknown'),
                score=getattr(chunk, 'score', 0.0),
                rerank_score=getattr(chunk, "rerank_score", None),
                content=getattr(chunk, 'chunk_text', ''),
                rank=i + 1
            ))

        metrics = self._calculate_metrics(search_results, expected_doc_id)

        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            expected_doc_id=expected_doc_id,
            search_results=search_results,
            metrics=metrics,
            search_time=search_time
        )


    def _calculate_metrics(self,
                           search_results: List[SearchResult],
                           expected_doc_id: str) -> Dict[str, float]:
        """
        Вычисление метрик качества поиска.
        """
        metrics = {}

        # Находим позицию релевантного документа
        relevant_rank = None
        for i, result in enumerate(search_results):
            if result.doc_id == expected_doc_id:
                relevant_rank = i + 1
                break

        # Precision@k, Recall@k, F1@k
        for k in self.k_values:
            top_k_results = search_results[:k]

            precision = self._calculate_precision(top_k_results, expected_doc_id)
            recall = self._calculate_recall(top_k_results, expected_doc_id)
            f1 = self._calculate_f1(precision, recall)

            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1

        metrics['mrr'] = self._calculate_mrr(relevant_rank)

        metrics['ndcg'] = self._calculate_ndcg(search_results, expected_doc_id)

        return metrics

    @staticmethod
    def _calculate_precision(results: List[SearchResult],
                             expected_doc_id: str) -> float:
        """Вычисление Precision."""
        if not results:
            return 0.0

        relevant_count = sum(1 for result in results if result.doc_id == expected_doc_id)
        return relevant_count / len(results)

    @staticmethod
    def _calculate_recall(results: List[SearchResult],
                          expected_doc_id: str) -> float:
        """Вычисление Recall."""
        relevant_found = any(result.doc_id == expected_doc_id for result in results)
        return 1.0 if relevant_found else 0.0

    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """Вычисление F1-меры."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _calculate_mrr(relevant_rank: Optional[int]) -> float:
        """Вычисление Reciprocal Rank."""
        if relevant_rank is None:
            return 0.0
        return 1.0 / relevant_rank

    @staticmethod
    def _calculate_ndcg(results: List[SearchResult],
                        expected_doc_id: str,
                        k: int = 10) -> float:
        """Вычисление nDCG."""
        if not results:
            return 0.0

        results = results[:k]

        ideal_gains = [1.0] + [0.0] * (len(results) - 1)

        actual_gains = []
        for result in results:
            if result.doc_id == expected_doc_id:
                actual_gains.append(1.0)
            else:
                actual_gains.append(0.0)

        dcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(actual_gains))

        idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal_gains))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def _aggregate_metrics(metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:
        """Агрегация метрик по всем запросам."""
        aggregated = {}

        for metric_name, values in metrics_dict.items():
            if values:
                aggregated[f"mean_{metric_name}"] = np.mean(values)
                aggregated[f"std_{metric_name}"] = np.std(values)
                aggregated[f"min_{metric_name}"] = np.min(values)
                aggregated[f"max_{metric_name}"] = np.max(values)
            else:
                aggregated.update({
                    f"mean_{metric_name}": 0.0,
                    f"std_{metric_name}": 0.0,
                    f"min_{metric_name}": 0.0,
                    f"max_{metric_name}": 0.0
                })

        return aggregated