from difflib import SequenceMatcher

from src.elastic.models import PositionModel
from src.pipeline.quote_locator import QuoteLocator

locator = QuoteLocator()

class Reranker:
    def __init__(self,
                 weight_semantic: float = 0.85,
                 weight_token: float = 0.10,
                 weight_sequence: float = 0.05):
        self.weight_semantic = weight_semantic
        self.weight_token = weight_token
        self.weight_sequence = weight_sequence


    @staticmethod
    def token_overlap(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        inter = sa.intersection(sb)
        return len(inter) / max(len(sa), len(sb)) if max(len(sa), len(sb)) > 0 else 0.0

    @staticmethod
    def sequence_score(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def rerank(self, query: str, result):
        chunks = result.chunks

        for chunk in chunks:
            text = chunk.chunk_text or ""

            token_score = self.token_overlap(query, text)
            seq_score = self.sequence_score(query, text)
            sem_score = chunk.score or 0.0  # vector score

            rerank_score = (
                    self.weight_semantic * sem_score +
                    self.weight_token * token_score +
                    self.weight_sequence * seq_score
            )

            chunk.rerank_score = rerank_score

            pos = locator.locate(query, text)
            chunk.keyword_positions = (
                [PositionModel(**p) for p in pos] if pos else None
            )

        # сортировка
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)

        result.chunks = chunks
        return result