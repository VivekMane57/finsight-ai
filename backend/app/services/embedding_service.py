from sentence_transformers import SentenceTransformer

class EmbeddingService:

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    @classmethod
    def create_embeddings(cls, chunks):

        return cls.model.encode(
            chunks,
            show_progress_bar=True
        )