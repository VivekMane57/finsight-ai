from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_documents(data: QueryRequest):

    sources = SearchService.search(data.query)

    answer = LLMService.generate_answer(
        query=data.query,
        sources=sources
    )

    return {
        "query": data.query,
        "answer": answer,
        "retrieval_method": "hybrid_faiss_bm25_with_citations",
        "sources_used": len(sources),
        "sources": sources
    }