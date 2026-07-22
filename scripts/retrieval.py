"""
retrieval.py
Shared helper: given a query review, return the top-K most relevant chunks
from the knowledge base built by 03_build_knowledge_base.py.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_index():
    with open(OUT_DIR / "kb_index.pkl", "rb") as f:
        return pickle.load(f)


def retrieve(query: str, index: dict, k_policy: int = 2, k_examples: int = 3):
    """Return top-k policy chunks and top-k labeled example reviews, separately,
    so the prompt can present them under clearly different headings."""
    chunks = index["chunks"]

    if index["backend"] == "tfidf":
        q_vec = index["vectorizer"].transform([query])
        sims = cosine_similarity(q_vec, index["matrix"]).flatten()
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(index["model_name"])
        q_emb = model.encode([query], normalize_embeddings=True)
        sims = (index["embeddings"] @ q_emb.T).flatten()

    order = np.argsort(-sims)

    policy_hits, example_hits = [], []
    for i in order:
        c = chunks[i]
        if c["type"] == "policy" and len(policy_hits) < k_policy:
            policy_hits.append({**c, "score": float(sims[i])})
        elif c["type"] == "example" and len(example_hits) < k_examples:
            example_hits.append({**c, "score": float(sims[i])})
        if len(policy_hits) >= k_policy and len(example_hits) >= k_examples:
            break

    return policy_hits, example_hits
