import faiss
import numpy as np

class FaissStore:

    @staticmethod
    def build_index(embeddings):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(
            embeddings
        )

        return index