import pickle
import faiss
import numpy as np

from backend.app.services.embedding_service import EmbeddingService


class SearchService:

    @staticmethod
    def search(query, top_k=5):

        with open("vectorstore/faiss_index.pkl", "rb") as f:
            index = pickle.load(f)

        with open("vectorstore/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        query_embedding = EmbeddingService.model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:
            results.append(chunks[idx])

        return results