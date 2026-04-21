import logging
import os

import urllib3
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.elastic")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD  = os.getenv("ELASTIC_PASSWORD")

client = Elasticsearch(
    [settings.es_host],
    basic_auth=(settings.elastic_user, settings.elastic_password),
    verify_certs=False,
    request_timeout=30,
    retry_on_timeout=True
)

def create_index():
    try:
        if client.indices.exists(index=settings.index_name):
            return

        mapping = {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "document_name": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "chunk_text": {"type": "text"},
                    "total_chunks": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": settings.vector_size,
                        "similarity": "cosine",
                        "index": True,
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100
                        },
                    },
                    "reg_number": {"type": "keyword"},
                    "reg_date": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm:ss"}
                }
            }
        }

        client.indices.create(index=settings.index_name, body=mapping)
        logger.info("Index created.")
    except Exception as e:
        logger.error(e)