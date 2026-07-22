"""
04_rag_copilot.py
The centerpiece: a retrieval-augmented LLM that classifies a review as
truthful/deceptive AND explains why, grounded in (a) retrieved policy
red-flags and (b) retrieved labeled example reviews (few-shot).

Runs against a LOCAL model via Ollama (https://ollama.com) — no API key,
no per-token cost. Install once, then:

    ollama pull llama3.2:3b        # ~2GB, runs fine on CPU
    python scripts/04_rag_copilot.py --review "Absolutely perfect stay..."

Also supports --mode zero_shot (no retrieval, for the ablation comparison
in evaluate.py) and --mode retrieve_only (skip the LLM call entirely and
just print what would be retrieved — useful for testing without Ollama
installed yet).
"""
import argparse
import json
import textwrap
import urllib.request

from retrieval import load_index, retrieve

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a trust & safety investigator's assistant. You
review a single hotel review and decide whether it is TRUTHFUL or
DECEPTIVE (paid/fabricated). You are given retrieved red-flag policy
notes and retrieved labeled example reviews for calibration. Ground your
verdict explicitly in the retrieved evidence — cite which red flag(s)
apply or which example the review resembles. Respond ONLY as JSON:
{"verdict": "truthful"|"deceptive", "confidence": 0-1, "reasoning": "..."}
"""


def build_prompt(review: str, policy_hits, example_hits) -> str:
    policy_block = "\n".join(f"- {h['text']}" for h in policy_hits) or "(none retrieved)"
    example_block = "\n".join(
        f'- [{h["label"].upper()}] "{h["text"][:200]}..."' for h in example_hits
    ) or "(none retrieved)"

    return textwrap.dedent(f"""
        RETRIEVED POLICY NOTES:
        {policy_block}

        RETRIEVED LABELED EXAMPLES:
        {example_block}

        REVIEW TO ASSESS:
        "{review}"

        Return your JSON verdict now.
    """).strip()


def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def classify(review: str, mode: str = "rag") -> dict:
    index = load_index()

    if mode == "zero_shot":
        prompt = f'REVIEW TO ASSESS:\n"{review}"\n\nReturn your JSON verdict now.'
        raw = call_ollama(prompt, system=SYSTEM_PROMPT.split("You are given")[0] +
                           'Respond ONLY as JSON: {"verdict": "truthful"|"deceptive", '
                           '"confidence": 0-1, "reasoning": "..."}')
        return json.loads(raw)

    policy_hits, example_hits = retrieve(review, index)

    if mode == "retrieve_only":
        return {"policy_hits": policy_hits, "example_hits": example_hits}

    prompt = build_prompt(review, policy_hits, example_hits)
    raw = call_ollama(prompt)
    result = json.loads(raw)
    result["retrieved_policy"] = [h["source"] for h in policy_hits]
    result["retrieved_examples"] = [h["label"] for h in example_hits]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--mode", choices=["rag", "zero_shot", "retrieve_only"], default="rag")
    args = parser.parse_args()

    result = classify(args.review, mode=args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
