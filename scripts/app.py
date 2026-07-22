"""
app.py — Streamlit demo UI for the Trust & Safety RAG Copilot.

    pip install streamlit
    streamlit run scripts/app.py

Paste a review, see: baseline ML verdict, retrieved evidence, and the
RAG-grounded LLM verdict with its reasoning, side by side.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import load_index, retrieve

import importlib
rag_copilot = importlib.import_module("04_rag_copilot")
api_copilot = importlib.import_module("06_api_copilot")

st.set_page_config(page_title="Trust & Safety RAG Copilot", layout="wide")
st.title("🔍 Trust & Safety RAG Copilot")
st.caption("Fake review triage — TF-IDF baseline + retrieval-augmented LLM, "
           "trained/evaluated on the Ott et al. Deceptive Opinion Spam Corpus.")

engine = st.sidebar.radio(
    "Model engine",
    ["Local (Ollama, free)", "Hosted API (costs money)"],
)
api_provider = None
if engine.startswith("Hosted"):
    api_provider = st.sidebar.selectbox("Provider", ["anthropic", "openai"])
    st.sidebar.caption("Reads ANTHROPIC_API_KEY / OPENAI_API_KEY from your environment.")

review = st.text_area("Paste a review to assess", height=150,
                       placeholder="This hotel was absolutely amazing, the best stay of my life...")

col1, col2 = st.columns(2)

if st.button("Analyze", type="primary") and review.strip():
    index = load_index()
    policy_hits, example_hits = retrieve(review, index)

    with col1:
        st.subheader("Retrieved evidence")
        st.markdown("**Policy red flags**")
        for h in policy_hits:
            st.markdown(f"> {h['text'][:300]}...")
        st.markdown("**Similar labeled examples**")
        for h in example_hits:
            badge = "🔴 deceptive" if h["label"] == "deceptive" else "🟢 truthful"
            st.markdown(f"{badge} — {h['text'][:200]}...")

    with col2:
        st.subheader("LLM verdict (RAG-grounded)")
        try:
            if engine.startswith("Hosted"):
                result = api_copilot.classify(review, provider=api_provider, mode="rag")
            else:
                result = rag_copilot.classify(review, mode="rag")
            verdict = result.get("verdict", "?")
            conf = result.get("confidence", 0)
            emoji = "🚩" if verdict == "deceptive" else "✅"
            st.markdown(f"### {emoji} {verdict.upper()}  (confidence: {conf})")
            st.write(result.get("reasoning", ""))
        except Exception as e:
            hint = ("Couldn't reach a local Ollama server. Install Ollama, run "
                     "`ollama pull llama3.2:3b`, then `ollama serve`, and try again."
                     if not engine.startswith("Hosted") else
                     f"Couldn't reach the {api_provider} API. Check that your API key "
                     "is set in the environment this app was launched from.")
            st.warning(f"{hint}\n\nDetails: {e}")

st.divider()
st.caption("Data: Ott, Choi, Cardie & Hancock (2011/2013), Deceptive Opinion Spam Corpus. "
           "Not affiliated with any marketplace's official trust & safety tooling — portfolio project.")
