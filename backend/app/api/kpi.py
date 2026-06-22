from fastapi import APIRouter
import json

from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService

router = APIRouter()


@router.get("/financial-kpis")
def financial_kpis():

    query = """
    revenue gross profit operating profit net profit operating margin
    net margin EPS earnings per share operating cash flow free cash flow
    total assets total liabilities fiscal year financial highlights
    """

    sources = SearchService.search(
        query=query,
        top_k=10
    )

    raw_kpis = LLMService.extract_kpis(
        sources=sources
    )

    try:
        kpis = json.loads(raw_kpis)
    except Exception:
        kpis = {
            "company": "Unknown",
            "fiscal_year": "Unknown",
            "currency": "Unknown",
            "revenue": None,
            "revenue_growth": None,
            "gross_profit": None,
            "operating_profit": None,
            "net_profit": None,
            "operating_margin": None,
            "net_margin": None,
            "eps_basic": None,
            "operating_cash_flow": None,
            "free_cash_flow": None,
            "total_assets": None,
            "total_liabilities": None,
            "raw_response": raw_kpis
        }

    return {
        "analysis_type": "auto_kpi_extraction",
        "kpis": kpis,
        "sources_used": len(sources),
        "sources": sources
    }