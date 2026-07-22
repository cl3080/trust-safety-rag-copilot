# Trust & Safety RAG Copilot

A fake-review detection and investigator-assist tool combining a classical
ML baseline with a retrieval-augmented local LLM. Built as a portfolio
project connecting fraud/risk experience with GenAI tooling.

## What it does
Given a marketplace review, the system:
1. Scores it with a TF-IDF + Logistic Regression baseline (fast, cheap,
   fully interpretable via top n-grams).
2. Retrieves the most relevant fraud-policy red flags and similar
   *labeled* example reviews from a small knowledge base.
3. Prompts a local LLM (via [Ollama](https://ollama.com), no API key) to
   render a verdict grounded in that retrieved evidence, with an auditable
   explanation — instead of an opaque label.

## Why this design
- **RAG over fine-tuning**: policy thresholds and red flags change; a
  fine-tuned classifier goes stale, but a retrievable knowledge base can
  be updated independently of the model.
- **Local model, not an API**: zero marginal cost, runs fully offline —
  relevant for a domain (fraud investigation) where sending user-generated
  content to a third-party API isn't always acceptable.
- **Baseline first**: the TF-IDF model is the sanity check and latency
  floor. The LLM/RAG layer is judged against it, not in a vacuum.

## Data
[Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus)
(Ott, Choi, Cardie & Hancock, ACL 2011 / NAACL 2013) — 1,600 hotel reviews,
balanced 50/50 truthful vs. deceptive (deceptive reviews were paid-for via
Mechanical Turk). `data/deceptive-opinion.csv` is included in this repo.
Please cite the original papers if you reuse the dataset.

## Setup
```bash
pip install -r requirements.txt

# option A — local model (free, offline, no key):
# 1. install Ollama: https://ollama.com/download
# 2. ollama pull llama3.2:3b
# 3. ollama serve   (if not already running as a background service)

# option B — hosted API model (costs money, needs a key):
export ANTHROPIC_API_KEY="sk-ant-..."     # or OPENAI_API_KEY for --provider openai
```

## Run it
```bash
python scripts/01_data_prep.py            # loads data, train/test split
python scripts/02_baseline_model.py        # TF-IDF + LogReg baseline
python scripts/03_build_knowledge_base.py  # builds the retrieval index

# local model:
python scripts/04_rag_copilot.py --review "Absolutely perfect stay, best hotel ever!!!"

# hosted API model:
python scripts/06_api_copilot.py --review "Absolutely perfect stay, best hotel ever!!!" --provider anthropic

# baseline vs zero-shot LLM vs RAG, side by side — pick the engine:
python scripts/05_evaluate.py --engine ollama --n_samples 60
python scripts/05_evaluate.py --engine api --api_provider anthropic --n_samples 20

# interactive demo:
streamlit run scripts/app.py
```

`04_rag_copilot.py --mode retrieve_only` works without Ollama installed —
useful for testing the retrieval layer on its own.

## Local vs. API: which to use

| | Local (`04_rag_copilot.py`, Ollama) | API (`06_api_copilot.py`, Claude/GPT) |
|---|---|---|
| Cost | Free after the one-time model download | Per-token — a full eval run has a real dollar cost |
| Data | Never leaves your machine | Sent to a third party over the network |
| Setup | Install Ollama, pull a model (~2GB) | Just an API key |
| Capability | Smaller model, more variance in JSON formatting / reasoning quality | Frontier-model reasoning, more reliable structured output |
| Latency | Slower on CPU, no rate limits | Fast, but subject to provider rate limits |
| Best framing | "I can run this fully offline for a data-sensitive domain" | "I can integrate with the same hosted APIs production systems use" |

Both scripts share the exact same `build_prompt` / retrieval logic from
`04_rag_copilot.py`, so a head-to-head comparison via `05_evaluate.py
--engine api` vs `--engine ollama` isolates the effect of model choice,
not prompt differences. Worth having both in your back pocket for an
interview — some teams are local/on-prem for compliance reasons, most
default to a hosted API.

## Results (baseline, this repo)
TF-IDF + Logistic Regression on the held-out test set (n=320):

| Metric | Value |
|---|---|
| Accuracy | 89.1% |
| F1 | 0.891 |

This lines up with Ott et al.'s reported ~89% for n-gram-based models on
this corpus. **Run `05_evaluate.py` locally** (after setting up Ollama) to
get your own zero-shot-LLM and RAG numbers — don't put numbers on your
resume that you haven't actually reproduced.

## Project structure
```
scripts/
  01_data_prep.py           data load + train/test split
  02_baseline_model.py      TF-IDF + LogisticRegression baseline
  03_build_knowledge_base.py builds the TF-IDF (or embedding) retrieval index
  retrieval.py               shared top-k retrieval function
  04_rag_copilot.py         RAG-grounded LLM classifier (calls local Ollama)
  06_api_copilot.py         same, but calls a hosted API (Anthropic/OpenAI)
  05_evaluate.py             baseline vs zero-shot vs RAG comparison — either engine
  app.py                     Streamlit demo UI
policy_docs/                 red-flag heuristics + regulatory context (retrieved as evidence)
data/deceptive-opinion.csv   the Ott et al. corpus
outputs/                     generated: splits, metrics, index, results
```

## Extending it
- Swap TF-IDF retrieval for `sentence-transformers` embeddings:
  `python scripts/03_build_knowledge_base.py --embeddings`
  (`retrieval.py` already supports both backends transparently.)
- Swap the review-fraud domain for anything else with labeled text pairs —
  the retrieval + grounded-verdict pattern is domain-agnostic.
- Add a second retrieval collection for real platform policy documents
  once you have access to them.

## Resume line (once you've run it and have your own RAG numbers)
> Built a retrieval-augmented LLM system for fake-review detection,
> combining a TF-IDF baseline (89% accuracy) with a local-LLM RAG copilot
> that grounds verdicts in retrieved fraud-policy evidence and labeled
> examples — improving [accuracy/interpretability] over zero-shot
> prompting while running entirely on local, no-API-cost infrastructure.
