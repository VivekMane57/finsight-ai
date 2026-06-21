import pickle
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Store:

    @staticmethod
    def build_index(chunks):
        tokenized_chunks = [
            chunk.lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)

        with open("vectorstore/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

        return bm25

    @staticmethod
    def search(query, top_k=5):
        with open("vectorstore/bm25_index.pkl", "rb") as f:
            bm25 = pickle.load(f)

        with open("vectorstore/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [chunks[i] for i in top_indices]