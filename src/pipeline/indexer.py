import logging
from queue import Queue
from threading import Thread
from elasticsearch import helpers

from src.config import settings
from src.elastic.client import client as elastic_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Indexer(Thread):
    def __init__(self, input_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.daemon = True

    def run(self):
        while True:
            batch = self.input_queue.get()
            if batch is None:
                self.input_queue.put(None)
                break

            try:
                actions = [
                    {
                        "_index": settings.index_name,
                        "_source": {
                            "document_id": c.document_id,
                            "document_name": c.document_name,
                            "file_name": c.file_name,
                            "chunk_text": c.chunk_text,
                            "total_chunks": c.total_chunks,
                            "chunk_index": c.chunk_index,
                            "reg_number": c.reg_number,
                            "reg_date": c.reg_date,
                            "vector": c.vector,
                        }
                    }
                    for c in batch
                ]
                helpers.bulk(elastic_client, actions, chunk_size=len(actions))
                logger.info(f"Батч из очереди записан в Elasticsearch (размер {len(batch)})")
            except Exception as e:
                logger.error(f"Ошибка при записи батча в Elasticsearch: {e}")
            finally:
                self.input_queue.task_done()