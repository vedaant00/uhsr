import numpy as np

def compute_cosine_similarity(query_embedding, embeddings):
    """Compute cosine similarity between query and document embeddings."""
    dot_product = np.dot(embeddings, query_embedding)
    norm_embeddings = np.linalg.norm(embeddings, axis=-1)
    norm_query = np.linalg.norm(query_embedding)
    return dot_product / (norm_embeddings * norm_query + 1e-8)

def compute_euclidean_similarity(query_embedding, embeddings):
    """Compute Euclidean similarity (1 / (1 + Euclidean distance))."""
    if np.ndim(embeddings) == 1:
        embeddings = embeddings.reshape(1, -1)

    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    return 1 / (1 + distances)

def compute_mahalanobis_similarity(query_embedding, embeddings):
    """Compute Mahalanobis similarity (normalized)."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    cov = np.cov(embeddings.T)
    regularization = 1e-5 * np.eye(cov.shape[0])
    cov += regularization 

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    diff = embeddings - query_embedding
    distances = np.array([np.sqrt(np.dot(np.dot(d, inv_cov), d.T)) for d in diff])
    similarities = 1 / (1 + distances)
    return np.nan_to_num(similarities, nan=0.0)

def compute_manhattan_similarity(query_embedding, embeddings):
    """Compute Manhattan similarity (1 / (1 + L1 Distance))."""
    if np.ndim(embeddings) == 1:
        embeddings = embeddings.reshape(1, -1)
    distances = np.sum(np.abs(embeddings - query_embedding), axis=1)
    return 1 / (1 + distances)

def compute_chebyshev_similarity(query_embedding, embeddings):
    """Compute Chebyshev similarity (1 / (1 + max absolute distance))."""
    if np.ndim(embeddings) == 1:
        embeddings = embeddings.reshape(1, -1)
    distances = np.max(np.abs(embeddings - query_embedding), axis=1)
    return 1 / (1 + distances)

def compute_jaccard_similarity(query_embedding, embeddings):
    """Compute Jaccard similarity for binary embeddings."""
    query_bin = query_embedding > 0
    embeddings_bin = embeddings > 0

    if np.ndim(embeddings_bin) == 1:
        embeddings_bin = embeddings_bin.reshape(1, -1)

    intersection = np.logical_and(query_bin, embeddings_bin).sum(axis=1)
    union = np.logical_or(query_bin, embeddings_bin).sum(axis=1)
    return intersection / (union + 1e-8) 

def compute_hamming_similarity(query_embedding, embeddings):
    """Compute Hamming similarity for binary vectors."""
    query_bin = query_embedding > 0
    embeddings_bin = embeddings > 0

    if np.ndim(embeddings_bin) == 1:
        embeddings_bin = embeddings_bin.reshape(1, -1)

    hamming_distances = np.sum(query_bin != embeddings_bin, axis=1)
    return 1 / (1 + hamming_distances)

def compute_semantic_similarity(query_embedding, embeddings, metric='cosine'):
    """
    Wrapper function to compute similarity based on the chosen metric.

    Parameters:
        query_embedding (np.array): The query embedding vector.
        embeddings (np.array): The document embeddings (shape: [N, d]).
        metric (str): The similarity metric to use.

    Returns:
        np.array: Array of similarity scores.

    Supported metrics:
    - 'cosine' (Cosine similarity)
    - 'euclidean' (Inverse Euclidean distance)
    - 'mahalanobis' (Mahalanobis similarity)
    - 'manhattan' (Inverse Manhattan distance)
    - 'chebyshev' (Inverse Chebyshev distance)
    - 'jaccard' (Jaccard similarity for binary embeddings)
    - 'hamming' (Hamming similarity for binary embeddings)
    """
    metric_functions = {
        'cosine': compute_cosine_similarity,
        'euclidean': compute_euclidean_similarity,
        'mahalanobis': compute_mahalanobis_similarity,
        'manhattan': compute_manhattan_similarity,
        'chebyshev': compute_chebyshev_similarity,
        'jaccard': compute_jaccard_similarity,
        'hamming': compute_hamming_similarity
    }

    if metric not in metric_functions:
        raise ValueError(f"❌ Unsupported metric: {metric}. Choose from {list(metric_functions.keys())}")

    return metric_functions[metric](query_embedding, embeddings)
