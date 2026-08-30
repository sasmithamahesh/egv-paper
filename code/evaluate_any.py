import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(result_file: str, name: str = "Results"):
    print("=" * 65)
    print(f"Evaluation: {name}")
    print("=" * 65)

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # 1. Verdict Distribution
    print("\n1. Verdict Distribution")
    print(df["verdict"].value_counts())
    print()

    # 2. Performance Breakdown
    right = df[df["answer_type"] == "right_answer"]
    hallu = df[df["answer_type"] == "hallucinated_answer"]

    supported_right = (right["verdict"] == "Supported").sum()
    caught_hallu = ((hallu["verdict"] == "Contradicted") | 
                    (hallu["verdict"] == "Partially Supported")).sum()

    print("2. Performance Breakdown")
    print(f"Right answers marked Supported        : {supported_right} / {len(right)}  ({supported_right/len(right)*100:.1f}%)")
    print(f"Hallucinated answers caught           : {caught_hallu} / {len(hallu)}  ({caught_hallu/len(hallu)*100:.1f}%)")
    print()

    # 3. Binary Metrics (same definition as before)
    def map_to_binary(row):
        if row["answer_type"] == "right_answer":
            return 1 if row["verdict"] == "Supported" else 0
        else:
            return 1 if row["verdict"] in ["Contradicted", "Partially Supported"] else 0

    df["binary_correct"] = df.apply(map_to_binary, axis=1)
    y_true = [1 if x == "right_answer" else 0 for x in df["answer_type"]]
    y_pred = df["binary_correct"].tolist()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("3. Binary Metrics (Supported vs Not for right answers)")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print()

    # 4. Abstention
    abstention = (df["verdict"] == "Unverifiable").mean() * 100
    print(f"4. Abstention Rate (Unverifiable) : {abstention:.1f}%")
    print()

    # 5. Cross table
    print("5. Detailed Counts")
    print(pd.crosstab(df["answer_type"], df["verdict"]))
    print("=" * 65)

    # Save summary
    summary = {
        "name": name,
        "total_claims": len(df),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "abstention_rate": round(abstention, 1),
        "supported_on_right": int(supported_right),
        "caught_hallucinated": int(caught_hallu),
        "verdict_distribution": df["verdict"].value_counts().to_dict()
    }

    out_name = Path(result_file).stem + "_metrics.json"
    with open(f"results/{out_name}", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved → results/{out_name}\n")
    return summary

if __name__ == "__main__":
    # Evaluate hybrid results
    if Path("results/stage3_hybrid.json").exists():
        evaluate("results/stage3_hybrid.json", "Hybrid Stage 3")

    # Evaluate lexical baseline
    if Path("results/baseline_lexical.json").exists():
        evaluate("results/baseline_lexical.json", "Lexical Baseline")

    # Also support older files if they exist
    if Path("results/stage3_results_v2.json").exists():
        evaluate("results/stage3_results_v2.json", "Previous NLI v2")