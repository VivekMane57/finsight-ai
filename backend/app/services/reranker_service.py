from sentence_transformers import CrossEncoder


class RerankerService:
    model = None
    available = False

    @staticmethod
    def _load_model():
        if RerankerService.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            RerankerService.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            RerankerService.available = True

        except Exception as e:
            print(f"CrossEncoder unavailable. Using fallback ranking. Reason: {e}")
            RerankerService.model = None
            RerankerService.available = False

    @staticmethod
    def rerank(query: str, documents: list[dict], top_k: int = 5):
        if not documents:
            return []

        RerankerService._load_model()

        if not RerankerService.available:
            return documents[:top_k]

        pairs = [(query, doc["chunk"]) for doc in documents]
        scores = RerankerService.model.predict(pairs)

        reranked_docs = []

        for doc, score in zip(documents, scores):
            updated_doc = doc.copy()
            updated_doc["rerank_score"] = float(score)
            updated_doc["rerank_method"] = "cross_encoder"
            reranked_docs.append(updated_doc)

        reranked_docs.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_docs[:top_k]