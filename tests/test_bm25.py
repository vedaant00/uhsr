from uhsr.bm25 import BM25

def test_bm25_scoring():
    docs = ["the cat sat on the mat", "the dog barked loudly"]
    bm25 = BM25(docs)
    score_cat = bm25.score("cat", 0)
    score_dog = bm25.score("dog", 1)
    assert score_cat > 0
    assert score_dog > 0

def test_bm25_search_top_result():
    docs = ["the quick brown fox", "jumps over the lazy dog"]
    bm25 = BM25(docs)
    indices, _, _ = bm25.search("fox", k=1)
    assert indices[0] == 0
