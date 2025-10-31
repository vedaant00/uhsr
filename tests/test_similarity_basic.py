import numpy as np
from uhsr.similarity import compute_semantic_similarity

def test_cosine_similarity():
    q = np.array([1, 0])
    docs = np.array([[1, 0], [0, 1]])
    sims = compute_semantic_similarity(q, docs, metric="cosine")
    assert sims[0] > sims[1]
    assert np.all((sims >= 0) & (sims <= 1))

def test_euclidean_similarity():
    q = np.array([1, 2])
    docs = np.array([[1, 2], [3, 4]])
    sims = compute_semantic_similarity(q, docs, metric="euclidean")
    assert np.isclose(sims[0], 1.0)
    assert sims[1] < sims[0]

def test_manhattan_similarity():
    q = np.array([0, 0])
    docs = np.array([[0, 0], [5, 5]])
    sims = compute_semantic_similarity(q, docs, metric="manhattan")
    assert sims[0] > sims[1]
