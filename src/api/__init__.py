from fastapi import APIRouter

from src.api.docs import router as docs_router

router = APIRouter()

router.include_router(docs_router)