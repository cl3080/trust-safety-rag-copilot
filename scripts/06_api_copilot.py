"""
06_api_copilot.py
Same RAG-grounded verdict as 04_rag_copilot.py, but calls a hosted API
model (Anthropic Claude or OpenAI) instead of a local Ollama model.

Why this exists alongside the local version: most companies you'd
interview with are running LLM features against a hosted API, not a
laptop-local model — being able to speak to both is worth more than
speaking to only one. This script also makes for a fair local-vs-API
head-to-head (see README: "Local vs. API tradeoffs").

Setup:
    pip install requests   # already in requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."      # for --provider anthropic
    export OPENAI_API_KEY="sk-..."             # for --provider openai

Usage:
    python scripts/06_api_copilot.py --review "Absolutely perfect stay!" --provider anthropic
    python scripts/06_api_copilot.py --review "Absolutely perfect stay!" --provider openai

Costs real money per call (unlike 04_rag_copilot.py) — see README for
the ballpark cost of a full evaluation run before you point 05_evaluate.py
at this instead of Ollama.
"""
import argparse
import json
import os
import urllib.request

from retrieval import load_index, retrieve
from importlib import import_module

# reuse the exact same prompt-building logic as the local version, so the
# two are a fair comparison — only the model call differs
_rag = import_module("04_rag_copilot")
build_prompt = _rag.build_prompt
SYSTEM_PROMPT = _rag.SYSTEM_PROMPT

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5"  # swap freely; this is just a default

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


def call_anthropic(prompt: str, system: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment")

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 400,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
    return text


def call_openai(prompt: str, system: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


CALLERS = {"anthropic": call_anthropic, "openai": call_openai}


def classify(review: str, provider: str = "anthropic", mode: str = "rag") -> dict:
    index = load_index()

    if mode == "zero_shot":
        prompt = f'REVIEW TO ASSESS:\n"{review}"\n\nReturn your JSON verdict now.'
    else:
        policy_hits, example_hits = retrieve(review, index)
        prompt = build_prompt(review, policy_hits, example_hits)

    json_instruction = (
        '\n\nRespond with ONLY valid JSON, no markdown fences: '
        '{"verdict": "truthful"|"deceptive", "confidence": 0-1, "reasoning": "..."}'
    )
    raw = CALLERS[provider](prompt, SYSTEM_PROMPT + json_instruction)

    # models occasionally wrap JSON in ```json fences despite instructions — strip defensively
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)
    if mode == "rag":
        result["retrieved_policy"] = [h["source"] for h in policy_hits]
        result["retrieved_examples"] = [h["label"] for h in example_hits]
    result["provider"] = provider
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--provider", choices=list(CALLERS), default="anthropic")
    parser.add_argument("--mode", choices=["rag", "zero_shot"], default="rag")
    args = parser.parse_args()

    result = classify(args.review, provider=args.provider, mode=args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
