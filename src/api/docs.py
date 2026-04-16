import io
import logging

import pandas as pd
from fastapi import UploadFile, File, APIRouter, HTTPException
from queue import Queue
import json
import time

from src.contracts.docs import SearchRequest, DeleteDocsRequest
from src.elastic.models import SearchParams
from src.elastic.retriever import Retriever
from src.pipeline.evaluator import Evaluator
from src.pipeline.models import DocumentSchema
from src.pipeline.producer import Producer
from src.pipeline.embedder import Embedder
from src.pipeline.indexer import Indexer
from src.pipeline.utils import split_into_batches
from src.pipeline.reranker import Reranker
from src.elastic.client import create_index
from src.services.gigachat_token_provider import GigaChatTokenProvider
from src.config import settings

router = APIRouter()

token_provider = GigaChatTokenProvider(
    auth_token=settings.gigachat_auth_token,
    auth_url=settings.gigachat_oauth_url,
    scope=settings.gigachat_scope,
    verify_ssl=False
)

embedder = Embedder(token_provider=token_provider)
retriever = Retriever()
reranker = Reranker()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@router.post('/api/docs', summary="Добавление документов.")
async def add_docs(file: UploadFile = File(
    ...,
           description=(
               "JSON-файл со списком документов.\n\n"
               "Корень файла: массив объектов **DocumentSchema**.\n\n"
               "DocumentSchema:\n"
               "  - ID: str\n"
               "  - Name: str\n"
               "  - RegNumber: str\n"
               "  - RegDate: datetime (ISO 8601)\n"
               "  - Files: List[FileSchema]\n\n"
               "FileSchema:\n"
               "  - Name: str\n"
               "  - Text: str\n"
           ))):
    """
    Ожидается JSON-файл следующего вида:

    ```json
    [
      {
        "ID": "doc-1",
        "Name": "Положение о внутреннем распорядке",
        "RegNumber": "123/АБ",
        "RegDate": "2024-01-10T00:00:00",
        "Files": [
          {
            "Name": "main",
            "Text": "Текст документа..."
          }
        ]
      }
    ]
    ```
    """
    try:
        create_index()

        documents = [DocumentSchema(**document) for document in json.loads(await file.read())]

        producer = Producer()
        chunks = producer.process(documents)
        total_chunks = len(chunks)

        batches = split_into_batches(chunks)
        total_batches = len(batches)

        start_time = time.time()

        batch_queue = Queue(maxsize=4)

        indexer = Indexer(batch_queue)
        indexer.start()

        logger.info(f"Начало эмбеддинга {total_chunks} чанков, размер батча: {embedder.batch_size}.")

        for idx, batch in enumerate(batches, 1):
            logger.info(f"Обработка батча {idx} / {total_batches} (чанков: {len(batch)}).")

            texts = [c.chunk_text.lower() for c in batch]
            vectors = embedder.embed_documents(texts)
            for chunk, vector in zip(batch, vectors):
                chunk.vector = vector

            batch_queue.put(batch)
            logger.info(f"Батч {idx} / {total_batches} добавлен в очередь.")

        elapsed_minutes = (time.time() - start_time) / 60
        logger.info(f"Эмбеддинг завершён, затрачено времени: {elapsed_minutes:.2f} минут.")

        all_chunks = [chunk for batch in batches for chunk in batch]
        return { "message": 'Ok.', "chunks": all_chunks[-10:] }
    except json.decoder.JSONDecodeError:
        return { "err": "Invalid JSON" }

@router.delete('/api/docs', summary="Удаление документов по списку ID.")
async def del_docs(body: DeleteDocsRequest):
    """
    Удаление документов по их ID.

    Пример тела запроса:

    ```json
    {
      "ids": ["doc-1", "doc-2", "doc-3"]
    }
    ```
    """
    try:
        result = retriever.delete_documents(body.ids)
        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.post('/api/docs/search', summary="Поиск релевантных чанков.")
async def search(request: SearchRequest):
    """
    Поиск релевантных чанков по текстовому запросу и (опционально) по диапазону дат.

    Пример тела запроса:

    ```json
    {
      "query": "положение о внутреннем распорядке",
      "date_from": "2024-01-01T00:00:00",
      "date_to": "2024-12-31T23:59:59",
      "size": 10
    }
    ```
    """
    query_embedding = embedder.embed_query(request.query)

    search_params = SearchParams(
        size=request.size,
        filter_date_from=request.date_from,
        filter_date_to=request.date_to
    )
    response = retriever.search(query_embedding, search_params)
    response = reranker.rerank(request.query, response)
    return response

@router.post('/api/docs/evaluate', summary="Оценка бенчмарка")
async def evaluate(file: UploadFile = File(
    ...,
    description=(
        "Excel-файл (.xlsx) с бенчмарком качества поиска.\n\n"
        "Каждая строка должна содержать запрос и ожидаемые релевантные документы.\n"
        "Состав колонок: number, question, answer, doc_name, doc_id"
    ),
)):
    contents = await file.read()
    excel_file = io.BytesIO(contents)

    df = pd.read_excel(excel_file)

    search_params = SearchParams(size=10)

    evaluator = Evaluator(retriever=retriever, k_values=[1, 5])

    results = evaluator.evaluate_benchmark(
        benchmark=df,
        embedder=embedder,
        search_params=search_params
    )

    return results