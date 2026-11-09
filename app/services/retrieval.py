from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.services.embeddings import EmbeddingService

class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def hybrid_search(
        self,
        query: str,
        db: AsyncSession,
        tenant_id: str = "default",
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Performs hybrid search combining pgvector cosine distance and PostgreSQL full-text search."""
        
        # 1. Generate query embedding
        query_vector = await self.embedding_service.generate_embedding(query)
        vector_str = f"[{','.join(str(x) for x in query_vector)}]"

        # 2. Hybrid SQL query utilizing pgvector (<-> cosine) and tsvector matching
        sql_query = text("""
            WITH vector_matches AS (
                SELECT 
                    id, 
                    document_id, 
                    content, 
                    metadata_json,
                    1 - (embedding <=> CAST(:vector_str AS vector)) AS vector_score
                FROM rag_document_chunks
                WHERE metadata_json->>'tenant_id' = :tenant_id
                ORDER BY embedding <=> CAST(:vector_str AS vector)
                LIMIT :fetch_limit
            ),
            text_matches AS (
                SELECT 
                    id,
                    ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', :raw_query)) AS text_score
                FROM rag_document_chunks
                WHERE metadata_json->>'tenant_id' = :tenant_id
                    AND to_tsvector('english', content) @@ plainto_tsquery('english', :raw_query)
                LIMIT :fetch_limit
            )
            SELECT 
                v.id,
                v.document_id,
                v.content,
                v.metadata_json,
                v.vector_score,
                COALESCE(t.text_score, 0.0) AS text_score,
                (:v_weight * v.vector_score + :t_weight * COALESCE(t.text_score, 0.0)) AS combined_score
            FROM vector_matches v
            LEFT JOIN text_matches t ON v.id = t.id
            ORDER BY combined_score DESC
            LIMIT :top_k;
        """)

        result = await db.execute(
            sql_query,
            {
                "vector_str": vector_str,
                "raw_query": query,
                "tenant_id": tenant_id,
                "fetch_limit": top_k * 3,
                "v_weight": vector_weight,
                "t_weight": text_weight,
                "top_k": top_k
            }
        )

        rows = result.mappings().all()
        
        return [
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "metadata": row["metadata_json"],
                "scores": {
                    "vector_score": float(row["vector_score"]),
                    "text_score": float(row["text_score"]),
                    "combined_score": float(row["combined_score"])
                }
            }
            for row in rows
        ]