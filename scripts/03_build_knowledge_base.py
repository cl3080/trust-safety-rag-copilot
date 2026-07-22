"""
03_build_knowledge_base.py
Builds the retrieval index for the RAG copilot. Two corpora are indexed:

  1. Policy docs (policy_docs/*.md) — red-flag heuristics + regulatory
     context, written for this project.
  2. Labeled example reviews (outputs/train.csv) — used for retrieval-
     augmented few-shot: given a new review, pull the K most similar
     *labeled* training reviews so the LLM has grounded examples of what
     "deceptive" and "truthful" look like in this exact domain.

Retrieval backend: TF-IDF + cosine similarity by default (zero extra
dependencies, runs anywhere). If sentence-transformers is installed,
pass --embeddings to use a real embedding model instead for better
semantic recall — see README for the tradeoffs.

Output: outputs/kb_index.pkl — loaded by 04_rag_copilot.py
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "policy_docs"
OUT_DIR = ROOT / "outputs"


def load_policy_chunks():
    """Split each policy doc into ~paragraph-sized chunks for retrieval."""
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text()
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para.split()) >= 15:  # skip headers / short fragments
                chunks.append({"source": path.name, "type": "policy", "text": para})
    return chunks


def load_example_reviews():
    train = pd.read_csv(OUT_DIR / "train.csv")
    examples = []
    for _, row in train.iterrows():
        label = "deceptive" if row["is_deceptive"] == 1 else "truthful"
        examples.append(
            {"source": f"train_example[{label}]", "type": "example", "label": label, "text": row["text"]}
        )
    return examples


def build_tfidf_index(chunks):
    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vec.fit_transform(texts)
    return {"backend": "tfidf", "vectorizer": vec, "matrix": matrix, "chunks": chunks}


def build_embedding_index(chunks):
    from sentence_transformers import SentenceTransformer  # optional dependency

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return {"backend": "embeddings", "model_name": "all-MiniLM-L6-v2", "embeddings": embeddings, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true", help="use sentence-transformers instead of TF-IDF")
    args = parser.parse_args()

    chunks = load_policy_chunks() + load_example_reviews()
    print(f"Indexing {len(chunks)} chunks "
          f"({sum(c['type'] == 'policy' for c in chunks)} policy, "
          f"{sum(c['type'] == 'example' for c in chunks)} labeled examples)")

    index = build_embedding_index(chunks) if args.embeddings else build_tfidf_index(chunks)

    with open(OUT_DIR / "kb_index.pkl", "wb") as f:
        pickle.dump(index, f)
    print(f"Saved outputs/kb_index.pkl (backend={index['backend']})")


if __name__ == "__main__":
    main()
