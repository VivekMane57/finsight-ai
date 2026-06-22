from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    vectorizer = None

    @staticmethod
    def create_embeddings(chunks):
        EmbeddingService.vectorizer = TfidfVectorizer(
            max_features=384,
            stop_words="english"
        )

        embeddings = EmbeddingService.vectorizer.fit_transform(chunks)
        return embeddings.toarray()

    @staticmethod
    def create_query_embedding(query):
        if EmbeddingService.vectorizer is None:
            raise ValueError("Vectorizer not initialized. Upload a document first.")

        embedding = EmbeddingService.vectorizer.transform([query])
        return embedding.toarray()