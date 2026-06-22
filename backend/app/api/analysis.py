from fastapi import APIRouter

from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService

router = APIRouter()


@router.post("/financial-summary")
def financial_summary():

    query = """
    financial highlights revenue total revenue sales operating margin
    profitability net profit net income earnings EPS balance sheet
    total assets total liabilities cash flow operating cash flow
    free cash flow debt borrowings liquidity business risks
    management discussion financial performance annual report
    """

    sources = SearchService.search(query=query, top_k=10)

    summary = LLMService.generate_financial_summary(
        sources=sources
    )

    return {
        "analysis_type": "financial_summary",
        "summary": summary,
        "sources_used": len(sources),
        "sources": sources
    }


@router.post("/credit-risk")
def credit_risk_analysis():

    query = """
    credit risk debt borrowings liabilities cash flow liquidity
    revenue profitability operating margin net profit working capital
    financial stability solvency repayment capacity business risks
    client concentration currency risk financial obligations
    """

    sources = SearchService.search(query=query, top_k=10)

    credit_report = LLMService.generate_credit_risk_analysis(
        sources=sources
    )

    return {
        "analysis_type": "credit_risk",
        "credit_report": credit_report,
        "sources_used": len(sources),
        "sources": sources
    }