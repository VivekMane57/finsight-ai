import pickle
import numpy as np

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.bm25_store import BM25Store
from backend.app.services.reranker_service import RerankerService


class SearchService:

    @staticmethod
    def search(query, top_k=5):

        with open("vectorstore/faiss_index.pkl", "rb") as f:
            index = pickle.load(f)

        with open("vectorstore/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        query_embedding = EmbeddingService.create_query_embedding(query)
        query_embedding = np.array(query_embedding).astype("float32")

        candidate_k = max(top_k * 4, 20)

        distances, indices = index.search(query_embedding, candidate_k)

        faiss_results = []

        for idx in indices[0]:
            if idx != -1 and idx < len(chunks):
                faiss_results.append(chunks[idx])

        bm25_results = BM25Store.search(query, candidate_k)

        hybrid_chunks = []
        seen = set()

        for chunk in faiss_results + bm25_results:
            if chunk not in seen:
                hybrid_chunks.append(chunk)
                seen.add(chunk)

        candidate_sources = []

        for i, chunk in enumerate(hybrid_chunks, start=1):
            candidate_sources.append(
                {
                    "source_id": i,
                    "chunk": chunk
                }
            )

        reranked_sources = RerankerService.rerank(
            query=query,
            documents=candidate_sources,
            top_k=top_k
        )

        final_sources = []

        for i, source in enumerate(reranked_sources, start=1):
            final_sources.append(
                {
                    "source_id": i,
                    "chunk": source["chunk"],
                    "rerank_score": source.get("rerank_score")
                }
            )

        return final_sources