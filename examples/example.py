from sentence_transformers import SentenceTransformer
from uhsr import UHSR
import numpy as np

# Sample dataset
documents = [
    "Apple releases new iPhone",
    "Tesla's stock price surges",
    "Google announces AI updates",
    "Amazon introduces drone delivery",
    "Microsoft acquires a gaming company"
]

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute document embeddings (normalized)
embeddings = model.encode(documents, normalize_embeddings=True)
# Compute query embedding
query_embedding = model.encode(["What did Tesla announce?"], normalize_embeddings=True)[0]

# Initialize UHSR retrieval system
retrieval_system = UHSR(documents, embeddings)

# Choose similarity metric: 'cosine', 'euclidean', or 'mahalanobis'
chosen_metric = 'cosine'

# Retrieve top 3 documents
retrieved_docs, scores = retrieval_system.retrieve("What did Tesla announce?", query_embedding, top_k=3, metric=chosen_metric)

print("🔹 Retrieved Documents using", chosen_metric, "similarity (Scores in [0,1]):")
for doc, score in zip(retrieved_docs, scores):
    print(f" - {doc} (Score: {score:.4f})")
