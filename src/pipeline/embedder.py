import logging
import time
from typing import List
import requests


from src.config import settings

class Embedder:
    def __init__(self, token_provider):
        self.token_provider = token_provider
        self.batch_size = settings.batch_size


    def _encode_local(self, texts: List[str]) -> List[List[float]]:
        # kwargs = {
        #    "batch_size": self.batch_size,
        #    "convert_to_numpy": True,
        #    "normalize_embeddings": True,
        #    "show_progress_bar": False,
        #}
        #encoded = self.model.encode(texts, **kwargs)
        #return encoded.tolist()
        pass


    def _encode_gigachat(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        Делает запрос вида:

        POST https://gigachat.devices.sberbank.ru/api/v1/embeddings
        {
          "model": "Embeddings",
          "input": ["...", "..."]
        }

        И ждёт ответ формата:
        {
          "object": "list",
          "data": [
            {"object": "embedding", "embedding": [...], "index": 0},
            {"object": "embedding", "embedding": [...], "index": 1}
          ],
          "model": "Embeddings"
        }
        """
        for attempt in range(1, max_retries + 1):
            try:
                token = self.token_provider.get_token()
                payload = {
                    "model": settings.gigachat_model,
                    "input": texts,
                }
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                resp = requests.post(
                    settings.gigachat_api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    verify=False
                )
                if resp.status_code == 401:
                    # токен умер - принудительно обновить и повторить
                    self.token_provider.get_token()
                    continue
                resp.raise_for_status()
                data = resp.json()
                # data["data"] список объектов { "object": "embedding", "embedding": [...], "index": n }
                items = data["data"]

                # На всякий случай отсортируем по index, чтобы порядок точно совпадал с исходным списком texts.
                items_sorted = sorted(items, key=lambda x: x["index"])

                embeddings = [item["embedding"] for item in items_sorted]
                return embeddings

            except Exception as e:
                logging.error(
                    "GigaChat error (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    e
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.0 * attempt)
        return []

    def _encode(self, texts: List[str], prompt_name: str) -> List[List[float]]:
        prefix = {
            "search_query": "search_query: ",
            "search_document": "search_document: ",
        }[prompt_name]

        prefixed = [prefix + t for t in texts]

        return self._encode_gigachat(prefixed)


    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        """
        Возвращает список векторов - один вектор на документ.
        Формат одинаковый для local и GigaChat.
        """
        return self._encode(texts, prompt_name="search_document")


    def embed_query(self, text: str) -> List[float]:
        """
        Возвращает один вектор (List[float]) для одного текста-запроса.
        Для GigaChat под капотом тоже уходит в /embeddings с одним input.
        """
        return self._encode([text], prompt_name="search_query")[0]