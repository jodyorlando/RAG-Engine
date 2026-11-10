from fastapi import APIRouter
from app.api.v1.endpoints import documents, search, rag

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["Documents Ingestion"])
api_router.include_router(search.router, prefix="/search", tags=["Hybrid Retrieval Search"])
api_router.include_router(rag.router, prefix="/rag", tags=["Grounded RAG Agent Engine"])