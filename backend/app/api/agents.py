from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.agents.financial_graph import financial_graph

router = APIRouter()


class AgentQueryRequest(BaseModel):
    query: str


@router.post("/financial-intelligence")
def run_financial_agents(data: AgentQueryRequest):

    result = financial_graph.invoke({
        "query": data.query,
        "sources": [],
        "summary": "",
        "credit_risk": "",
        "final_answer": ""
    })

    return {
        "query": data.query,
        "agent_workflow": [
            "retrieval_agent",
            "summary_agent",
            "credit_risk_agent",
            "final_analyst_agent"
        ],
        "final_answer": result["final_answer"],
        "sources_used": len(result["sources"]),
        "sources": result["sources"]
    }