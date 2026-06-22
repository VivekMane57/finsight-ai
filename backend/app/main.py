from fastapi import FastAPI

from backend.app.api.documents import router as documents_router
from backend.app.api.chat import router as chat_router
from backend.app.api.analysis import router as analysis_router
from backend.app.api.kpi import router as kpi_router
from backend.app.api.agents import router as agents_router
from backend.app.api.investment import router as investment_router

app = FastAPI(
    title="FinSight AI",
    version="1.0.0"
)

app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"]
)

app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"]
)

app.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Financial Analysis"]
)

app.include_router(
    kpi_router,
    prefix="/kpi",
    tags=["KPI Dashboard"]
)

app.include_router(
    agents_router,
    prefix="/agents",
    tags=["LangGraph Agents"]
)

app.include_router(
    investment_router,
    prefix="/analysis",
    tags=["Investment Analysis"]
)

@app.get("/")
def root():
    return {
        "status": "running",
        "project": "FinSight AI"
    }