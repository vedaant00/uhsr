import math
from collections import Counter

class BM25:
    """
    Implements the BM25 lexical scoring algorithm for text retrieval.

    BM25 is an extension of TF-IDF that incorporates term frequency saturation and document length normalization.
    It assigns scores to documents based on their relevance to a given query.

    Attributes:
        original_documents (list): The list of full-text documents.
        documents (list): Tokenized documents (each document as a list of words).
        k1 (float): BM25 parameter controlling term frequency saturation.
        b (float): BM25 parameter controlling document length normalization.
        N (int): Number of documents in the corpus.
        avgdl (float): Average document length across all documents.
        idf (dict): Precomputed inverse document frequencies (IDF) for terms.
        doc_lengths (list): Precomputed document lengths for normalization.
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        """
        Initializes the BM25 ranking model.

        Parameters:
            documents (list of str): List of textual documents to be indexed.
            k1 (float): Controls term frequency saturation (default: 1.5).
            b (float): Controls document length normalization (default: 0.75).
        """
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
        Computes the Inverse Document Frequency (IDF) for all terms in the corpus.

        Returns:
            dict: A mapping from term to its IDF score.
        """
        df = Counter(word for doc in self.documents for word in set(doc))
        return {word: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1) for word, freq in df.items()}

    def score(self, query, doc_idx):
        """
        Computes the BM25 relevance score of a document with respect to a query.

        Parameters:
            query (str): The search query.
            doc_idx (int): The index of the document to be scored.

        Returns:
            float: BM25 score indicating document relevance.
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
        Retrieves the top-k most relevant documents for a given query.

        Parameters:
            query (str): The search query.
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
