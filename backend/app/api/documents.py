from fastapi import APIRouter, UploadFile, File
import shutil
import os
import pickle

from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.chunking import ChunkingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.faiss_store import FaissStore

router = APIRouter()

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = DocumentProcessor.extract_pdf_text(file_path)

    chunks = ChunkingService.create_chunks(extracted_text)

    embeddings = EmbeddingService.create_embeddings(chunks)

    index = FaissStore.build_index(embeddings)

    index_path = os.path.join(VECTORSTORE_DIR, "faiss_index.pkl")
    chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return {
        "filename": file.filename,
        "characters": len(extracted_text),
        "chunks": len(chunks),
        "embedding_dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
        "status": "indexed"
    }