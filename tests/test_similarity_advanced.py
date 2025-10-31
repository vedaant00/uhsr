import numpy as np
import pytest
from uhsr.similarity import compute_semantic_similarity

def test_jaccard_similarity_binary_vectors():
    q = np.array([1, 0, 1])
    docs = np.array([[1, 0, 1], [1, 1, 0]])
    sims = compute_semantic_similarity(q, docs, metric="jaccard")
    assert np.all((sims >= 0) & (sims <= 1))
    assert sims[0] > sims[1]

def test_hamming_similarity_binary_vectors():
    q = np.array([1, 0, 1, 0])
    docs = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    sims = compute_semantic_similarity(q, docs, metric="hamming")
    assert sims[0] > sims[1]
    assert np.all((sims > 0) & (sims <= 1))

def test_invalid_metric_raises_error():
    q = np.array([1, 2, 3])
    docs = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        compute_semantic_similarity(q, docs, metric="unknown_metric")
