import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    vectorizer = None

    VECTORIZER_PATH = "vectorstore/tfidf_vectorizer.pkl"

    @staticmethod
    def create_embeddings(chunks):
        os.makedirs("vectorstore", exist_ok=True)

        EmbeddingService.vectorizer = TfidfVectorizer(
            max_features=384,
            stop_words="english"
        )

        embeddings = EmbeddingService.vectorizer.fit_transform(chunks)

        joblib.dump(
            EmbeddingService.vectorizer,
            EmbeddingService.VECTORIZER_PATH
        )

        return embeddings.toarray()

    @staticmethod
    def create_query_embedding(query):
        if EmbeddingService.vectorizer is None:
            if os.path.exists(EmbeddingService.VECTORIZER_PATH):
                EmbeddingService.vectorizer = joblib.load(
                    EmbeddingService.VECTORIZER_PATH
                )
            else:
                raise ValueError(
                    "Vectorizer not initialized. Please upload and process a document first."
                )

        embedding = EmbeddingService.vectorizer.transform([query])

        return embedding.toarray()