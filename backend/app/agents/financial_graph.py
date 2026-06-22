from typing import TypedDict, List, Dict

from langgraph.graph import StateGraph, END

from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService


class FinancialAgentState(TypedDict):
    query: str
    sources: List[Dict]
    summary: str
    credit_risk: str
    final_answer: str


def retrieval_agent(state: FinancialAgentState):
    query = state["query"]

    sources = SearchService.search(
        query=query,
        top_k=10
    )

    state["sources"] = sources
    return state


def summary_agent(state: FinancialAgentState):
    summary = LLMService.generate_financial_summary(
        sources=state["sources"]
    )

    state["summary"] = summary
    return state


def credit_risk_agent(state: FinancialAgentState):
    credit_risk = LLMService.generate_credit_risk_analysis(
        sources=state["sources"]
    )

    state["credit_risk"] = credit_risk
    return state


def final_analyst_agent(state: FinancialAgentState):
    final_answer = f"""
# FinSight AI Multi-Agent Financial Intelligence Report

## User Query
{state["query"]}

---

## Financial Summary Agent Output

{state["summary"]}

---

## Credit Risk Agent Output

{state["credit_risk"]}

---

## Source-Grounded Evidence

Sources Used: {len(state["sources"])}
"""

    state["final_answer"] = final_answer
    return state


def build_financial_graph():
    graph = StateGraph(FinancialAgentState)

    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("summary_agent", summary_agent)
    graph.add_node("credit_risk_agent", credit_risk_agent)
    graph.add_node("final_analyst_agent", final_analyst_agent)

    graph.set_entry_point("retrieval_agent")

    graph.add_edge("retrieval_agent", "summary_agent")
    graph.add_edge("summary_agent", "credit_risk_agent")
    graph.add_edge("credit_risk_agent", "final_analyst_agent")
    graph.add_edge("final_analyst_agent", END)

    return graph.compile()


financial_graph = build_financial_graph()