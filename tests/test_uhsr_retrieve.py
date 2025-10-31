import numpy as np
from uhsr.core import UHSR

def test_retrieve_basic(monkeypatch):
    # mock FAISS retrieval class
    class MockFAISS:
        def __init__(self, embeddings):
            pass
        def search(self, query_embedding, k=5):
            return np.arange(k), np.linspace(0.9, 0.5, k)
    monkeypatch.setattr("uhsr.core.FAISSRetrieval", MockFAISS)

    docs = ["hello world", "the quick brown fox", "lorem ipsum"]
    embeddings = np.random.rand(3, 4)
    model = UHSR(docs, embeddings)

    results, scores = model.retrieve("hello", embeddings[0], top_k=2)
    assert len(results) == 2
    assert all(isinstance(x, str) for x in results)
    assert all(isinstance(s, np.floating) for s in scores)
