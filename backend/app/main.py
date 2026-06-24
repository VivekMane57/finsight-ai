from fastapi import FastAPI

from backend.app.api.documents import router as documents_router
from backend.app.api.chat import router as chat_router
from backend.app.api.analysis import router as analysis_router
from backend.app.api.kpi import router as kpi_router
from backend.app.api.agents import router as agents_router
from backend.app.api.investment import router as investment_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.ragas import router as ragas_router
from backend.app.api.credit_ml import router as credit_router

app = FastAPI(
    title="FinSight AI",
    version="1.0.0"
)

# Documents
app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"]
)

# Chat
app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"]
)

# Financial Analysis
app.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Financial Analysis"]
)

# KPI Dashboard
app.include_router(
    kpi_router,
    prefix="/kpi",
    tags=["KPI Dashboard"]
)

# LangGraph Agents
app.include_router(
    agents_router,
    prefix="/agents",
    tags=["LangGraph Agents"]
)

# Investment Analysis
app.include_router(
    investment_router,
    prefix="/investment",
    tags=["Investment Analysis"]
)

# Custom LLM Evaluation
app.include_router(
    evaluation_router,
    prefix="/evaluation",
    tags=["LLM Evaluation"]
)

# RAGAS Evaluation
app.include_router(
    ragas_router,
    prefix="/ragas",
    tags=["RAGAS Evaluation"]
)

# Credit Risk Analysis
app.include_router(
    credit_router,
    prefix="/credit",
    tags=["Credit Risk ML"]
)

@app.get("/")
def root():
    return {
        "status": "running",
        "project": "FinSight AI",
        "features": [
            "Hybrid RAG",
            "CrossEncoder Reranking",
            "Financial Summary",
            "Credit Risk Analysis",
            "Investment Analysis",
            "KPI Dashboard",
            "LangGraph Agents",
            "LLM Evaluation",
            "RAGAS Evaluation"
        ]
    }