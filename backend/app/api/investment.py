from fastapi import APIRouter
from backend.app.agents.investment_agent import InvestmentAgent

router = APIRouter()


@router.post("/investment-analysis")
def investment_analysis():
    return InvestmentAgent.analyze()