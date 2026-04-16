from typing import Optional, List, Dict
import re

from rapidfuzz import fuzz


class QuoteLocator:
    def __init__(self, threshold: float = 90.0, min_word_len: int = 3):
        self.threshold = threshold
        self.min_word_len = min_word_len

    def locate(self, query: str, text: str) -> Optional[List[Dict]]:
        if not query or not text:
            return None

        text_lower = text.lower()

        # Разбиваем запрос на слова
        query_words = re.findall(r"\w+", query.lower())

        # Оставляем только более-менее содержательные
        candidate_words = {w for w in query_words if len(w) >= self.min_word_len}
        if not candidate_words:
            return None

        keywords: List[Dict] = []
        seen_spans = set()

        # Для каждого слова считаем fuzz.partial_ratio
        for word in candidate_words:
            score = fuzz.partial_ratio(word, text_lower)
            if score < self.threshold:
                continue

            # Если слово релевантно ищем его точные вхождения в тексте
            pattern = re.compile(r"\b" + re.escape(word) + r"\b",
                                 re.IGNORECASE | re.UNICODE)

            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if (start, end) in seen_spans:
                    continue
                seen_spans.add((start, end))

                keywords.append(
                    {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                    }
                )

        if not keywords:
            return None

        keywords.sort(key=lambda k: k["start"])
        return keywords