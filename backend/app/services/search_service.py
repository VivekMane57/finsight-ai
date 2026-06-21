import pickle
import numpy as np

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.bm25_store import BM25Store


class SearchService:

    @staticmethod
    def search(query, top_k=5):

        with open("vectorstore/faiss_index.pkl", "rb") as f:
            index = pickle.load(f)

        with open("vectorstore/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        query_embedding = EmbeddingService.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = index.search(query_embedding, top_k)

        faiss_results = [chunks[idx] for idx in indices[0]]
        bm25_results = BM25Store.search(query, top_k)

        hybrid_results = []
        seen = set()

        for chunk in faiss_results + bm25_results:
            if chunk not in seen:
                hybrid_results.append(chunk)
                seen.add(chunk)

        final_results = hybrid_results[:top_k]

        sources = []
        for i, chunk in enumerate(final_results, start=1):
            sources.append({
                "source_id": i,
                "chunk": chunk
            })

        return sources