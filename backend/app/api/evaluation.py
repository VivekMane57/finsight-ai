from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.evaluation.rag_evaluator import RAGEvaluator
from backend.app.evaluation.llm_judge import LLMJudge
from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService

router = APIRouter()


class EvaluationRequest(BaseModel):
    query: str


@router.post("/rag")
def evaluate_rag(data: EvaluationRequest):
    return RAGEvaluator.evaluate_query(data.query)


@router.post("/judge")
def llm_as_judge(data: EvaluationRequest):
    sources = SearchService.search(
        query=data.query,
        top_k=5
    )

    answer = LLMService.generate_answer(
        query=data.query,
        sources=sources
    )

    judge_result = LLMJudge.evaluate_answer(
        query=data.query,
        answer=answer,
        sources=sources
    )

    return {
        "query": data.query,
        "answer": answer,
        "judge_result": judge_result,
        "sources": sources
    }