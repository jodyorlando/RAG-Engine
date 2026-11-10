import pytest

@pytest.mark.asyncio
async def test_health_check(async_client):
    """Test health check route."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

@pytest.mark.asyncio
async def test_search_empty_query_validation(async_client):
    """Test empty search query validation."""
    response = await async_client.post("/api/v1/search/query", json={"query": "   ", "tenant_id": "default"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."

@pytest.mark.asyncio
async def test_rag_stream_empty_query_validation(async_client):
    """Test empty streaming query validation."""
    response = await async_client.post("/api/v1/rag/stream", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."