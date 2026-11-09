from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database.session import get_db
from app.services.retrieval import RetrievalService

router = APIRouter()
retrieval_service = RetrievalService()

class SearchRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = "default"
    top_k: Optional[int] = 5

@router.post("/query")
async def search_documents(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Executes hybrid vector + full-text search against uploaded document chunks."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = await retrieval_service.hybrid_search(
        query=payload.query,
        db=db,
        tenant_id=payload.tenant_id or "default",
        top_k=payload.top_k or 5
    )

    return {
        "status": "success",
        "query": payload.query,
        "results_count": len(results),
        "results": results
    }