import openai
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

class Reranker:
    """
    Implements a reranking model for improving retrieval accuracy.
    
    Supports:
    - OpenAI Reranker
    - Cohere Rerank
    - SentenceTransformer Cross-Encoder
    - Custom user-provided models
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", openai_api_key=None, reranker_type="huggingface"):
        """
        Initialize the reranker.

        Parameters:
            model_name (str): The model name for reranking (if using Hugging Face models).
            openai_api_key (str): OpenAI API Key (if using OpenAI reranker).
            reranker_type (str): The type of reranker: 'huggingface', 'openai', or 'custom'.
        """
        self.reranker_type = reranker_type
        self.openai_api_key = openai_api_key

        if reranker_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        elif reranker_type == "openai":
            if openai_api_key is None:
                raise ValueError("OpenAI API key is required for OpenAI Reranker.")

    def rerank(self, query, candidates):
        """
        Reranks retrieved documents based on query relevance.

        Parameters:
            query (str): The user query.
            candidates (list): List of retrieved documents.

        Returns:
            tuple: (ranked documents, sorted scores)
        """
        if self.reranker_type == "huggingface":
            return self._huggingface_rerank(query, candidates)
        elif self.reranker_type == "openai":
            return self._openai_rerank(query, candidates)

    def _huggingface_rerank(self, query, candidates):
        """
        Uses a Hugging Face cross-encoder to rerank candidates.

        Parameters:
            query (str): The query string.
            candidates (list): List of candidate documents.

        Returns:
            tuple: (sorted documents, sorted scores)
        """
        scores = []
        for doc in candidates:
            inputs = self.tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                score = self.model(**inputs).logits.item()
            scores.append(score)

        scores = 1 / (1 + np.exp(-np.array(scores)))

        sorted_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in sorted_indices], scores[sorted_indices]

    def _openai_rerank(self, query, candidates):
        """
        Uses OpenAI API to rerank documents.

        Parameters:
            query (str): The query string.
            candidates (list): List of candidate documents.

        Returns:
            tuple: (sorted documents, sorted scores)
        """
        openai.api_key = self.openai_api_key
        results = openai.Embedding.create(
            input=[query] + candidates,
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(results['data'][0]['embedding'])
        doc_embeddings = np.array([results['data'][i + 1]['embedding'] for i in range(len(candidates))])

        scores = np.dot(doc_embeddings, query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8)

        sorted_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in sorted_indices], scores[sorted_indices]
