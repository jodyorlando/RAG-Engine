from typing import List
from openai import AsyncOpenAI
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    async def generate_embedding(self, text: str) -> List[float]:
        """Generates a single vector embedding for a text string."""
        cleaned_text = text.replace("\n", " ")
        response = await self.client.embeddings.create(
            input=[cleaned_text],
            model=self.model
        )
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates vector embeddings for a batch of text chunks concurrently."""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        response = await self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model
        )
        return [data.embedding for data in response.data]