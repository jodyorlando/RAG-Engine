from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ForeignKey, func
from pgvector.sqlalchemy import Vector
from app.database.session import Base
from app.core.config import settings

class Document(Base):
    __tablename__ = "rag_documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    tenant_id = Column(String, default="default", index=True)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

class DocumentChunk(Base):
    __tablename__ = "rag_document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime, server_default=func.now())