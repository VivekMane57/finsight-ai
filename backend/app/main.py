from fastapi import FastAPI

from backend.app.api.documents import router as documents_router
from backend.app.api.chat import router as chat_router

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

@app.get("/")
def root():
    return {
        "status": "running",
        "project": "FinSight AI"
    }