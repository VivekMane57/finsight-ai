from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.search_service import SearchService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_documents(data: QueryRequest):

    chunks = SearchService.search(
        data.query
    )

    return {
        "query": data.query,
        "retrieved_chunks": chunks
    }