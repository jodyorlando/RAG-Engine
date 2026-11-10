from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

from app.database.session import get_db
from app.services.rag import RAGService

router = APIRouter()
rag_service = RAGService()

class RAGQueryRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = "default"
    top_k: Optional[int] = 4

@router.post("/stream")
async def stream_rag_response(
    payload: RAGQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Executes grounded RAG search and streams tokens + citation metadata using Server-Sent Events (SSE)."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        rag_service.generate_rag_stream(
            query=payload.query,
            db=db,
            tenant_id=payload.tenant_id or "default",
            top_k=payload.top_k or 4
        ),
        media_type="text/event-stream"
    )