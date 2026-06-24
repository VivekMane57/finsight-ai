from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService
from backend.app.evaluation.ragas_evaluator import (
    RagasEvaluator
)

router = APIRouter()


class RagasRequest(BaseModel):
    query: str


@router.post("/evaluate")
def evaluate_ragas(data: RagasRequest):

    sources = SearchService.search(
        query=data.query,
        top_k=5
    )

    answer = LLMService.generate_answer(
        data.query,
        sources
    )

    contexts = [
        source["chunk"]
        for source in sources
    ]

    scores = (
        RagasEvaluator.evaluate_response(
            question=data.query,
            answer=answer,
            contexts=contexts
        )
    )

    return {
        "query": data.query,
        "answer": answer,
        "scores": scores
    }