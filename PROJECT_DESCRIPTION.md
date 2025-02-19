# Unified Hyperbolic Spectral Retrieval (UHSR)

Unified Hyperbolic Spectral Retrieval (UHSR) is an advanced **hybrid text retrieval model** that seamlessly integrates **lexical search (BM25)** with **semantic search (FAISS/Pinecone)** while employing **spectral re-ranking** for **interpretable and normalized** relevance scores in the **[0,1] range**.

## 🚀 Key Features

- **🔍 Hybrid Retrieval:** Combines **BM25** for lexical scoring and **dense vector** semantic similarity for contextual understanding.
- **🎯 Multi-Metric Similarity:** Supports **cosine, euclidean, mahalanobis, manhattan, chebyshev, jaccard, and hamming** similarity.
- **🔬 Spectral Re-Ranking:** Uses **graph Laplacian & Fiedler vector** to boost highly relevant candidates.
- **⚡ AI-powered Reranking:** Supports **Hugging Face Cross-Encoders & OpenAI API-based Reranking**.
- **📈 Interpretable Scores:** Final relevance scores are **logistic-normalized** in **[0,1]** for **easy ranking**.
- **🚀 Scalable & Efficient:** Works with **FAISS (local)** for fast retrieval and **Pinecone (cloud-based)** for large-scale vector search.

---

## 🛠️ **How It Works**

UHSR **enhances traditional retrieval** by blending **BM25-based keyword matching** with **semantic vector representations** using the following pipeline:

| Step | Description |
|------|-------------|
| 1️⃣ **Lexical Filtering** | Uses **BM25** to rank documents by keyword relevance |
| 2️⃣ **Semantic Scoring** | Computes similarity using **FAISS or Pinecone** |
| 3️⃣ **Fusion Process** | Blends scores via **logistic normalization & harmonic fusion** |
| 4️⃣ **Spectral Re-Ranking** | Uses **graph Laplacian analysis** to boost central candidates |
| 5️⃣ **(Optional) AI Reranking** | Uses **OpenAI API or Hugging Face Cross-Encoders** |

---

## 🌍 **Supported Retrieval Methods**
- ✅ **BM25 (Lexical Matching)**
- ✅ **FAISS (Local Vector Search)**
- ✅ **Pinecone (Cloud Vector Search)**
- ✅ **Hugging Face Rerankers**
- ✅ **OpenAI API-based Reranking**

---

## 📌 **Why UHSR?**
- **Better Search Results:** Combines **exact keyword matching (BM25)** with **contextual embeddings (Semantic Search)**.
- **Faster & Scalable:** Uses **FAISS for local retrieval** or **Pinecone for cloud-based vector search**.
- **Interpretable Ranking:** Outputs **normalized scores in [0,1]**, making it easy to **interpret**.
- **Multi-Metric Similarity:** Supports **cosine, euclidean, mahalanobis, manhattan, chebyshev, jaccard, and hamming**.

---

## 🎯 **Intended Use**

UHSR is designed for:
- **Information Retrieval Research**
- **Search Engines & Recommendation Systems**
- **NLP Applications in AI & Machine Learning**
- **Academic & Industry-scale Document Ranking**

---

## 📂 **Code & Documentation**
For complete documentation, usage examples, and implementation details, visit the **[GitHub repository](https://github.com/vedaant00/uhsr).**

_Learn More about this package on [Medium](https://vedaantsingh706.medium.com/revolutionizing-text-retrieval-with-uhsr-a-hybrid-approach-combining-lexical-semantic-spectral-6c7e28c3e7d9)._

---

### 🔥 **Try UHSR today and revolutionize your search engine!** 🚀