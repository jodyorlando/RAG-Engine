from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, base_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        base_meta = base_metadata or {}
        docs = self.splitter.create_documents(texts=[text], metadatas=[base_meta])
        
        chunks = []
        for idx, doc in enumerate(docs):
            meta = doc.metadata.copy()
            meta["chunk_index"] = idx
            chunks.append({
                "chunk_index": idx,
                "content": doc.page_content,
                "metadata": meta
            })
        return chunks