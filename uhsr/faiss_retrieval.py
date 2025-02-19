import numpy as np
import faiss

class FAISSRetrieval:
    """
    Implements semantic retrieval using FAISS with an HNSW index.

    FAISS is used for efficient approximate nearest neighbor (ANN) search over large-scale vector embeddings.

    Attributes:
        dimension (int): Dimensionality of the vector embeddings.
        index (faiss.IndexHNSWFlat): FAISS index for fast ANN search.
    """

    def __init__(self, embeddings):
        """
        Initializes the FAISS index and adds precomputed embeddings.

        Parameters:
            embeddings (numpy.ndarray): 2D array of document embeddings.
        """
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        """
        Performs a nearest neighbor search using FAISS.

        Parameters:
            query_embedding (numpy.ndarray): The vector embedding of the query.
            k (int): Number of closest results to return.

        Returns:
            tuple: (indices, distances)
                indices: Indices of the top-k closest documents.
                distances: FAISS distances to the closest documents.
        """
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return indices[0], distances[0]
