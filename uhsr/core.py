import numpy as np
import faiss
import math
from collections import Counter

# ---------------------------
# Utility Functions
# ---------------------------
def logistic_norm(scores, a=5):
    """
    Apply logistic normalization to map scores into the (0,1) interval.
    
    Parameters:
        scores (np.array): Array of raw scores.
        a (float): Steepness parameter; higher values produce a steeper transition around the mean.
    
    Returns:
        np.array: Normalized scores in the range (0,1).
        
    The function centers scores around their mean and applies a logistic function:
        normalized_score = 1 / (1 + exp(-a * (score - mean(score))))
    """
    return 1 / (1 + np.exp(-a * (scores - np.mean(scores))))

# ---------------------------
# 1. BM25 Implementation for Lexical Search
# ---------------------------
class BM25:
    """
    Implements the BM25 lexical scoring algorithm.
    
    Attributes:
        original_documents (list): The list of full-text documents.
        documents (list): Tokenized documents (each document as a list of words).
        k1 (float): BM25 parameter controlling term frequency saturation.
        b (float): BM25 parameter controlling document length normalization.
        N (int): Number of documents.
        avgdl (float): Average document length.
        idf (dict): Inverse document frequencies for terms.
        doc_lengths (list): Lengths of each tokenized document.
    """
    def __init__(self, documents, k1=1.5, b=0.75):
        self.original_documents = documents
        self.documents = [doc.lower().split() for doc in documents]
        self.k1 = k1
        self.b = b
        self.N = len(self.documents)
        self.avgdl = sum(len(doc) for doc in self.documents) / self.N
        self.idf = self.compute_idf()
        self.doc_lengths = [len(doc) for doc in self.documents]

    def compute_idf(self):
        """
        Compute the inverse document frequency (idf) for each term.
        
        Returns:
            dict: A mapping from term to its idf score.
            
        Uses the BM25 idf formula:
            idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        """
        df = Counter(word for doc in self.documents for word in set(doc))
        return {word: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1) for word, freq in df.items()}

    def score(self, query, doc_idx):
        """
        Compute the BM25 score for a specific document with respect to a query.
        
        Parameters:
            query (str): The query string.
            doc_idx (int): The index of the document in the corpus.
        
        Returns:
            float: The BM25 score.
        """
        query_words = query.lower().split()
        score = 0
        for word in query_words:
            if word in self.idf:
                tf = self.documents[doc_idx].count(word)
                score += (self.idf[word] * tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avgdl))
        return score

    def search(self, query, k=5):
        """
        Search for the top-k documents based on BM25 scores.
        
        Parameters:
            query (str): The query string.
            k (int): The number of top documents to return.
        
        Returns:
            tuple: (candidate_indices, candidate_docs, candidate_scores)
                candidate_indices: List of indices of the top documents.
                candidate_docs: List of full-text documents.
                candidate_scores: List of BM25 scores for the top documents.
        """
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        candidate_indices = [i for i, _ in scores]
        candidate_docs = [self.original_documents[i] for i in candidate_indices]
        candidate_scores = [s for _, s in scores]
        return candidate_indices, candidate_docs, candidate_scores

# ---------------------------
# 2. FAISS-based Semantic Retrieval with ANN
# ---------------------------
class FAISSRetrieval:
    """
    Implements semantic retrieval using FAISS with HNSW index.
    
    Attributes:
        dimension (int): Dimensionality of the embeddings.
        index (faiss.IndexHNSWFlat): FAISS index for approximate nearest neighbor search.
    """
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        """
        Search for the top-k nearest neighbors using FAISS.
        
        Parameters:
            query_embedding (np.array): The query embedding vector.
            k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            tuple: (indices, distances) for the top-k neighbors.
        """
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return indices[0], distances[0]

# ---------------------------
# 3. Advanced Similarity Functions for Semantic Retrieval
# ---------------------------
def compute_semantic_similarity(query_embedding, embeddings, metric='cosine'):
    """
    Compute similarity scores between a query embedding and each document embedding.
    
    Parameters:
        query_embedding (np.array): The query embedding vector.
        embeddings (np.array): The document embeddings (shape: [N, d]).
        metric (str): The similarity metric to use: 'cosine', 'euclidean', or 'mahalanobis'.
    
    Returns:
        np.array: An array of similarity scores.
    """
    if metric == 'cosine':
        if np.ndim(embeddings) == 1:
            dot_product = np.dot(embeddings, query_embedding)
            norm_embeddings = np.linalg.norm(embeddings)
            norm_query = np.linalg.norm(query_embedding)
            return dot_product / (norm_embeddings * norm_query + 1e-8)
        else:
            dot_product = np.dot(embeddings, query_embedding)
            norm_embeddings = np.linalg.norm(embeddings, axis=1)
            norm_query = np.linalg.norm(query_embedding)
            return dot_product / (norm_embeddings * norm_query + 1e-8)
    elif metric == 'euclidean':
        if np.ndim(embeddings) == 1:
            distance = np.linalg.norm(embeddings - query_embedding)
            return 1 / (1 + distance)
        else:
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            return 1 / (1 + distances)
    elif metric == 'mahalanobis':
        cov = np.cov(embeddings.T)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        if np.ndim(embeddings) == 1:
            diff = embeddings - query_embedding
            distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
            return 1 / (1 + distance)
        else:
            diff = embeddings - query_embedding
            distances = np.array([np.sqrt(np.dot(np.dot(d, inv_cov), d.T)) for d in diff])
            return 1 / (1 + distances)
    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'euclidean', or 'mahalanobis'.")

# ---------------------------
# 4. Unified Hyperbolic Spectral Retrieval (UHSR)
# ---------------------------
class UHSR:
    """
    Unified Hyperbolic Spectral Retrieval (UHSR) fuses BM25 lexical scores and semantic
    similarity scores (using user-selectable metrics) using logistic normalization and a
    weighted fusion strategy. It further refines candidate rankings through spectral re-ranking.
    
    Attributes:
        documents (list): List of full-text documents.
        embeddings (np.array): Precomputed document embeddings.
        bm25 (BM25): BM25 instance for lexical scoring.
    """
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.bm25 = BM25(documents)

    def tanh_normalize(self, scores):
        """
        Apply hyperbolic tangent normalization to compress score ranges.
        
        Parameters:
            scores (np.array): Raw scores.
            
        Returns:
            np.array: Tanh-normalized scores.
        """
        return np.tanh(scores)

    def harmonic_fusion(self, s1, s2, eps=1e-8):
        """
        Fuse two scores using the harmonic mean.
        
        Parameters:
            s1 (np.array): First set of normalized scores.
            s2 (np.array): Second set of normalized scores.
            eps (float): Small constant to avoid division by zero.
        
        Returns:
            np.array: Fused scores, guaranteed to be in (0,1] if s1 and s2 are in (0,1].
        """
        return 2.0 / ((1.0 / (s1 + eps)) + (1.0 / (s2 + eps)))

    def build_similarity_graph(self, candidate_embeddings):
        """
        Build a similarity graph (matrix) for candidate embeddings using cosine similarity.
        
        Parameters:
            candidate_embeddings (np.array): Embeddings of candidate documents.
        
        Returns:
            np.array: A square similarity matrix.
        """
        n = candidate_embeddings.shape[0]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = compute_semantic_similarity(candidate_embeddings[i], candidate_embeddings[j], metric='cosine')
        return sim_matrix

    def spectral_rerank(self, candidate_indices, fusion_scores):
        """
        Apply spectral re-ranking on a candidate set using graph Laplacian centrality.
        
        Parameters:
            candidate_indices (np.array): Indices of candidate documents.
            fusion_scores (np.array): Fused scores of candidate documents.
        
        Returns:
            np.array: Normalized centrality scores in [0,1] for candidates.
        """
        candidate_embeddings = self.embeddings[candidate_indices]
        sim_matrix = self.build_similarity_graph(candidate_embeddings)
        degrees = np.sum(sim_matrix, axis=1)
        D = np.diag(degrees)
        L = D - sim_matrix
        eigvals, eigvecs = np.linalg.eigh(L)
        if len(eigvals) > 1:
            fiedler_vector = eigvecs[:, 1]
        else:
            fiedler_vector = np.ones(len(candidate_indices))
        centrality = np.abs(fiedler_vector)
        centrality_norm = logistic_norm(centrality, a=5)
        return centrality_norm

    def retrieve(self, query, query_embedding, top_k=5, metric='cosine', gamma=0.7):
        """
        Retrieve top documents for a query by fusing BM25 lexical and semantic similarity scores,
        and then re-ranking using spectral centrality.
        
        Parameters:
            query (str): The query string.
            query_embedding (np.array): Embedding of the query.
            top_k (int): Number of top documents to retrieve.
            metric (str): Similarity metric to use ('cosine', 'euclidean', or 'mahalanobis').
            gamma (float): Weight for fusion score versus centrality in final score (0<=gamma<=1).
        
        Returns:
            tuple: (final_documents, final_scores)
                final_documents (list): Ranked list of documents.
                final_scores (np.array): Final relevance scores in the range [0,1].
        """
        # A. BM25 lexical filtering
        bm25_indices, bm25_docs, bm25_scores = self.bm25.search(query, top_k)
        bm25_scores = np.array(bm25_scores)
        if len(bm25_scores) > 0:
            bm25_norm = logistic_norm(self.tanh_normalize(bm25_scores), a=5)
        else:
            bm25_norm = np.zeros(top_k)
        
        # B. Semantic scoring using the chosen metric on all docs, restricted to BM25 candidates
        all_semantic_scores = compute_semantic_similarity(query_embedding, self.embeddings, metric=metric)
        semantic_candidate_scores = all_semantic_scores[bm25_indices]
        semantic_norm = logistic_norm(self.tanh_normalize(semantic_candidate_scores), a=5)
        
        # C. Fusion: Weighted arithmetic mean of BM25 and semantic scores (both in [0,1])
        fusion_scores = 0.5 * bm25_norm + 0.5 * semantic_norm
        
        # D. Spectral re-ranking: Combine fusion score with centrality via weighted average.
        centrality = self.spectral_rerank(np.array(bm25_indices), fusion_scores)
        final_scores = gamma * fusion_scores + (1 - gamma) * centrality
        final_order = np.argsort(final_scores)[::-1]
        final_indices = np.array(bm25_indices)[final_order]
        final_scores = final_scores[final_order]
        
        return [self.documents[i] for i in final_indices], final_scores

# End of uhsr/core.py