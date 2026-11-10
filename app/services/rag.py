import json
from typing import AsyncGenerator, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.retrieval import RetrievalService

class RAGService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.retrieval_service = RetrievalService()

    async def generate_rag_stream(
        self,
        query: str,
        db: AsyncSession,
        tenant_id: str = "default",
        top_k: int = 4
    ) -> AsyncGenerator[str, None]:
        """Retrieves grounded document context and streams answer tokens via Server-Sent Events (SSE)."""

        # 1. Retrieve top-k document context chunks using your Day 3/4 hybrid search
        retrieved_chunks = await self.retrieval_service.hybrid_search(
            query=query,
            db=db,
            tenant_id=tenant_id,
            top_k=top_k
        )

        # 2. Format retrieved context & citations payload
        context_blocks = []
        citations = []
        for idx, item in enumerate(retrieved_chunks):
            citation_info = {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "filename": item["metadata"].get("filename", "unknown"),
                "combined_score": round(item["scores"]["combined_score"], 4)
            }
            citations.append(citation_info)
            context_blocks.append(f"--- Context Block [{idx + 1}] (Source: {citation_info['filename']}) ---\n{item['content']}")

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant document context found."

        # First SSE payload: Send citation metadata block to client immediately
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        # 3. Construct System & User Prompt
        system_prompt = (
            "You are an enterprise AI knowledge assistant. Answer the user's question using ONLY the provided document context blocks below. "
            "If the answer cannot be answered based on the context, politely state that the provided documents do not contain enough information. "
            "Do not fabricate facts outside the context."
        )

        user_prompt = f"Document Context:\n{context_str}\n\nUser Question: {query}"

        # 4. Stream response from OpenAI GPT-4o-mini
        response_stream = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            stream=True
        )

        async for chunk in response_stream:
            content = chunk.choices[0].delta.content if chunk.choices else ""
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        # Final SSE signal
        yield "data: [DONE]\n\n"