# Unified Hyperbolic Spectral Retrieval (UHSR)

Unified Hyperbolic Spectral Retrieval (UHSR) is a novel text retrieval algorithm that fuses lexical and semantic search methods into a unified, robust, and scalable system. UHSR combines BM25 lexical scoring with dense vector semantic similarity (supporting user-selectable metrics such as cosine, euclidean, or Mahalanobis) via advanced techniques including logistic normalization, harmonic fusion, and spectral re-ranking based on graph Laplacian analysis.

## Features

- **Hybrid Retrieval:** Integrates BM25 (lexical) and semantic (vector) search.
- **Advanced Fusion:** Applies logistic normalization to map scores into [0,1] and fuses them using a harmonic mean.
- **Spectral Re-Ranking:** Uses spectral analysis (Fiedler vector from the graph Laplacian) to boost candidates that are central in the candidate set.
- **Metric Flexibility:** Supports multiple semantic similarity metrics: cosine, euclidean, and Mahalanobis.
- **Scalable and Robust:** Designed to work with both small and large datasets using FAISS for fast approximate nearest neighbor search.
- **Interpretable Scores:** Final relevance scores are normalized to the [0,1] range.

## Novel Contributions

- **Unified Fusion Method:** Novel combination of BM25 and semantic scores through logistic normalization and harmonic fusion.
- **Spectral Re-Ranking:** Incorporates spectral analysis to enhance the ranking of candidates based on centrality.
- **User-Selectable Similarity Metrics:** Offers flexibility in choosing the similarity measure that best suits your data.
- **End-to-End Retrieval Pipeline:** From raw text to final ranked documents with interpretable scores.

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/uhsr-retrieval.git
cd uhsr-retrieval
pip install -r requirements.txt
```

*Note:* The package uses `faiss-cpu` by default. For GPU support, install with:

```bash
pip install uhsr-retrieval[gpu]
```

## Usage

Below is an example of how to use UHSR in your own Python code:

```python
from sentence_transformers import SentenceTransformer
from uhsr import UHSR
import numpy as np

# Sample dataset of documents
documents = [
    "Apple releases new iPhone",
    "Tesla's stock price surges",
    "Google announces AI updates",
    "Amazon introduces drone delivery",
    "Microsoft acquires a gaming company"
]

# Load an embedding model (e.g., all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute document embeddings (normalized)
embeddings = model.encode(documents, normalize_embeddings=True)

# Compute query embedding
query_embedding = model.encode(["What did Tesla announce?"], normalize_embeddings=True)[0]

# Initialize the UHSR retrieval system
retrieval_system = UHSR(documents, embeddings)

# Retrieve documents using cosine similarity (choose from 'cosine', 'euclidean', or 'mahalanobis')
retrieved_docs, scores = retrieval_system.retrieve("What did Tesla announce?", query_embedding, top_k=3, metric='cosine')

print("Retrieved Documents:")
for doc, score in zip(retrieved_docs, scores):
    print(f"{doc} (Score: {score:.4f})")
```

## Repository Structure

```
uhsr-retrieval/
├── uhsr/
│   ├── __init__.py         # Package initialization; imports from core.py
│   └── core.py             # Contains UHSR, BM25, FAISSRetrieval, and utility functions
├── examples/
│   └── example.py          # Example script demonstrating usage of UHSR
├── README.md               # This file
├── setup.py                # Packaging script for PyPI
└── requirements.txt        # List of dependencies
```

## Requirements

- numpy
- sentence-transformers
- FAISS (either `faiss-cpu` or `faiss-gpu`)

## License

This project is licensed under the MIT License.

## Conclusion

UHSR is a novel and unified text retrieval framework that effectively combines lexical and semantic signals through advanced normalization, fusion, and spectral re-ranking techniques. Its interpretable scoring and flexibility in choosing similarity metrics make it a strong candidate for further research and practical applications. Contributions and improvements from the community are welcome!

---