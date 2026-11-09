import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader
import io

from app.database.session import get_db
from app.models.document import Document, DocumentChunk
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService

router = APIRouter()
chunker = ChunkingService(chunk_size=1000, chunk_overlap=150)
embedding_service = EmbeddingService()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form("default"),
    db: AsyncSession = Depends(get_db)
):
    """Uploads a document (PDF/TXT), chunks it, embeds chunks via OpenAI, and stores in pgvector."""
    content_type = file.content_type or ""
    filename = file.filename or "unknown.txt"
    
    # 1. Extract raw text based on file format
    body = await file.read()
    if filename.endswith(".pdf") or "pdf" in content_type:
        pdf_reader = PdfReader(io.BytesIO(body))
        raw_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    else:
        raw_text = body.decode("utf-8", errors="ignore")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Document contains no extractable text.")

    # 2. Chunk text using recursive splitting
    chunks_data = chunker.split_text(raw_text, base_metadata={"filename": filename, "tenant_id": tenant_id})
    if not chunks_data:
        raise HTTPException(status_code=400, detail="Failed to create text chunks from document.")

    # 3. Create parent document record
    doc_id = str(uuid.uuid4())
    db_doc = Document(
        id=doc_id,
        filename=filename,
        file_type=filename.split(".")[-1] if "." in filename else "txt",
        tenant_id=tenant_id,
        total_chunks=len(chunks_data)
    )
    db.add(db_doc)

    # 4. Generate embeddings in batch
    texts_to_embed = [c["content"] for c in chunks_data]
    embeddings = await embedding_service.generate_embeddings_batch(texts_to_embed)

    # 5. Store chunks with embeddings in pgvector table
    for idx, (chunk, vector) in enumerate(zip(chunks_data, embeddings)):
        db_chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=chunk["chunk_index"],
            content=chunk["content"],
            embedding=vector,
            metadata_json=chunk["metadata"]
        )
        db.add(db_chunk)

    await db.commit()

    return {
        "status": "success",
        "document_id": doc_id,
        "filename": filename,
        "total_chunks_stored": len(chunks_data),
        "tenant_id": tenant_id
    }