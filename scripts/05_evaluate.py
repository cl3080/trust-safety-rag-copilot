"""
05_evaluate.py
Runs all approaches on a sample of the held-out test set and reports
accuracy/F1 side by side:
  1. baseline   — TF-IDF + Logistic Regression (already scored in 02)
  2. zero_shot  — LLM classifies with no retrieval context
  3. rag        — LLM classifies with retrieved policy + example evidence

Steps 2-3 can run against either engine:
  --engine ollama (default) — local model, free, needs Ollama running
  --engine api               — hosted model (Anthropic/OpenAI), costs money,
                                needs an API key (see 06_api_copilot.py)

This script defaults to a small --n_samples so a full run doesn't take
forever on CPU, or cost much on the API engine. Bump it up for a more
robust number once you've confirmed the pipeline works end to end.

Usage:
    # local, free:
    ollama pull llama3.2:3b && ollama serve &
    python scripts/05_evaluate.py --engine ollama --n_samples 60

    # hosted API, costs money — check pricing before raising --n_samples:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/05_evaluate.py --engine api --api_provider anthropic --n_samples 20
"""
import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# module names start with a digit, so they can't be `import`ed directly
ollama_copilot = importlib.import_module("04_rag_copilot")
api_copilot = importlib.import_module("06_api_copilot")

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_llm_pass(test_df: pd.DataFrame, mode: str, engine: str, api_provider: str) -> pd.DataFrame:
    preds = []
    for _, row in test_df.iterrows():
        try:
            if engine == "ollama":
                result = ollama_copilot.classify(row["text"], mode=mode)
            else:
                result = api_copilot.classify(row["text"], provider=api_provider, mode=mode)
            pred = 1 if result.get("verdict") == "deceptive" else 0
        except Exception as e:  # server not running, no API key, rate limit, etc.
            print(f"  [warn] {mode} call failed ({e}); recording as unknown (-1)")
            pred = -1
        preds.append(pred)
    test_df[f"{mode}_pred"] = preds
    return test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=60,
                         help="rows sampled from the test set for the LLM passes "
                              "(kept small: local CPU inference is slow, API calls cost money)")
    parser.add_argument("--skip_llm", action="store_true",
                         help="only report the baseline (use this if neither engine is set up yet)")
    parser.add_argument("--engine", choices=["ollama", "api"], default="ollama",
                         help="ollama = local, free. api = hosted model, costs money.")
    parser.add_argument("--api_provider", choices=["anthropic", "openai"], default="anthropic",
                         help="only used when --engine api")
    args = parser.parse_args()

    test = pd.read_csv(OUT_DIR / "test_with_baseline_preds.csv")
    baseline_acc = accuracy_score(test["is_deceptive"], test["baseline_pred"])
    baseline_f1 = f1_score(test["is_deceptive"], test["baseline_pred"])
    results = {"baseline": {"accuracy": baseline_acc, "f1": baseline_f1, "n": len(test)}}
    print(f"baseline   | n={len(test):>4} | acc={baseline_acc:.3f} | f1={baseline_f1:.3f}")

    if not args.skip_llm:
        engine_label = args.engine if args.engine == "ollama" else f"api:{args.api_provider}"
        sample = test.sample(n=min(args.n_samples, len(test)), random_state=42).copy()
        for mode in ["zero_shot", "rag"]:
            print(f"\nRunning {mode} ({engine_label}) over {len(sample)} sampled reviews...")
            sample = run_llm_pass(sample, mode, args.engine, args.api_provider)
            valid = sample[sample[f"{mode}_pred"] != -1]
            if len(valid):
                acc = accuracy_score(valid["is_deceptive"], valid[f"{mode}_pred"])
                f1 = f1_score(valid["is_deceptive"], valid[f"{mode}_pred"])
                results[f"{mode}_{engine_label}"] = {"accuracy": acc, "f1": f1, "n": len(valid)}
                print(f"{mode:10} | n={len(valid):>4} | acc={acc:.3f} | f1={f1:.3f}")
            else:
                hint = "is `ollama serve` running and is the model pulled?" if args.engine == "ollama" \
                    else "is your API key set correctly?"
                print(f"{mode}: no successful calls — {hint}")

        sample.to_csv(OUT_DIR / f"llm_eval_sample_{engine_label.replace(':', '_')}.csv", index=False)

    (OUT_DIR / "eval_summary.json").write_text(json.dumps(results, indent=2))
    print("\nSaved outputs/eval_summary.json")


if __name__ == "__main__":
    main()
