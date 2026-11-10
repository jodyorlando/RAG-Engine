# Enterprise Multimodal RAG & Knowledge Retrieval Engine

An asynchronous, microservice-based Multimodal Knowledge & Hybrid Retrieval Engine built with FastAPI, PostgreSQL (`pgvector`), OpenAI Embeddings, and Server-Sent Events (SSE). 

Engineered for production workloads requiring semantic vector search combined with exact full-text keyword ranking over unstructured document context.

---

## Key Architecture & Features

- **Asynchronous Pipeline:** Built on FastAPI, `asyncio`, and `sqlalchemy.ext.asyncio` with `asyncpg` drivers for non-blocking I/O.
- **Batch Vector Ingestion:** Extracts text from unstructured documents (PDFs, TXT), recursively chunks content, generates 1536-dimensional vector embeddings (`text-embedding-3-small`), and stores records transactionally in `pgvector`.
- **Hybrid Retrieval Engine:** Merges dense vector cosine similarity (`1 - (embedding <=> CAST(:vector AS vector))`) with PostgreSQL full-text keyword search (`tsvector` / `ts_rank_cd`), scoring candidates via weighted rank fusion.
- **Streaming Grounded Generation:** Streams GPT-4o-mini response tokens in real-time alongside exact chunk metadata citations using Server-Sent Events (`text/event-stream`).
- **Multi-Tenant Isolation:** Filters vector search scopes dynamically via `tenant_id` database parameters.

---

## Tech Stack

- **Framework:** Python 3.11, FastAPI, AsyncIO, Pydantic v2
- **Vector Storage:** PostgreSQL 16 (`pgvector` extension)
- **Database Driver:** SQLAlchemy 2.0 (Async Engine), AsyncPG
- **AI Models:** OpenAI `text-embedding-3-small` & `gpt-4o-mini`
- **Containerization:** Docker Compose, Dockerfile
- **Deployment:** Railway Cloud Infrastructure

---

## Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379/0