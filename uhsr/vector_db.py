import pinecone
import numpy as np

class PineconeVectorDB:
    """
    Pinecone-based Vector Database for storing and retrieving document embeddings.
    """

    def __init__(self, api_key, index_name="uhsr-index", dimension=384, metric="cosine"):
        """
        Initializes Pinecone and connects to the specified index.

        Parameters:
            api_key (str): Your Pinecone API key.
            index_name (str): Name of the Pinecone index.
            dimension (int): Embedding dimension.
            metric (str): Distance metric to use ('cosine', 'euclidean').
        """
        pinecone.init(api_key=api_key, environment="us-west1-gcp")
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension, metric=metric)
        self.index = pinecone.Index(index_name)

    def add_embeddings(self, doc_ids, embeddings):
        """
        Adds document embeddings to Pinecone.

        Parameters:
            doc_ids (list): List of document IDs.
            embeddings (numpy.ndarray): Array of document embeddings.
        """
        vectors = [(str(doc_id), embedding.tolist()) for doc_id, embedding in zip(doc_ids, embeddings)]
        self.index.upsert(vectors)

    def query(self, query_embedding, top_k=5):
        """
        Searches for the most relevant documents using Pinecone.

        Parameters:
            query_embedding (numpy.ndarray): Query embedding.
            top_k (int): Number of top matches to return.

        Returns:
            List of (doc_id, similarity_score) tuples.
        """
        query_result = self.index.query(queries=[query_embedding.tolist()], top_k=top_k, include_metadata=False)
        matches = query_result["matches"]
        return [(match["id"], match["score"]) for match in matches]
