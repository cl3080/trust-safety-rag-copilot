"""
02_baseline_model.py
Classical ML baseline: TF-IDF (word 1-2 grams) + Logistic Regression.
This is the reference point the LLM / RAG approaches in 03 and 04 are
measured against. Ott et al. (2011) report ~89% accuracy with n-gram +
SVM on the same corpus, so this should land in a similar range.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    train = pd.read_csv(OUT_DIR / "train.csv")
    test = pd.read_csv(OUT_DIR / "test.csv")

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words="english", sublinear_tf=True
    )
    X_train = vec.fit_transform(train["text"])
    X_test = vec.transform(test["text"])

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, train["is_deceptive"])

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(test["is_deceptive"], preds)
    f1 = f1_score(test["is_deceptive"], preds)
    report = classification_report(test["is_deceptive"], preds, target_names=["truthful", "deceptive"])
    cm = confusion_matrix(test["is_deceptive"], preds).tolist()

    print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}\n")
    print(report)
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)

    # top predictive n-grams (interpretability — useful for the "why flagged" narrative)
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_deceptive = [feature_names[i] for i in coefs.argsort()[-15:][::-1]]
    top_truthful = [feature_names[i] for i in coefs.argsort()[:15]]

    results = {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "top_deceptive_ngrams": top_deceptive,
        "top_truthful_ngrams": top_truthful,
    }
    (OUT_DIR / "baseline_results.json").write_text(json.dumps(results, indent=2))

    test = test.copy()
    test["baseline_pred"] = preds
    test["baseline_prob_deceptive"] = probs
    test.to_csv(OUT_DIR / "test_with_baseline_preds.csv", index=False)
    print("\nSaved outputs/baseline_results.json and outputs/test_with_baseline_preds.csv")


if __name__ == "__main__":
    main()
