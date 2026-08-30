import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ====================== LOAD RESULTS ======================
with open("results/stage3_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

df = pd.DataFrame(results)

print("=" * 65)
print("EGV Stage 3 - Full Evaluation Report")
print("=" * 65)

# ----------------------------------------------------------
# 1. Verdict Distribution
# ----------------------------------------------------------
print("\n1. Verdict Distribution (out of 600 claims)")
print(df["verdict"].value_counts())
print()

# ----------------------------------------------------------
# 2. Performance on Right vs Hallucinated Answers
# ----------------------------------------------------------
right = df[df["answer_type"] == "right_answer"]
hallu = df[df["answer_type"] == "hallucinated_answer"]

print("2. Performance Breakdown")
print(f"Right Answers marked Supported        : {(right['verdict'] == 'Supported').sum()} / 300  ({(right['verdict'] == 'Supported').mean()*100:.1f}%)")
print(f"Hallucinated Answers correctly caught : {((hallu['verdict'] == 'Contradicted') | (hallu['verdict'] == 'Partially Supported')).sum()} / 300  ({((hallu['verdict'] == 'Contradicted') | (hallu['verdict'] == 'Partially Supported')).mean()*100:.1f}%)")
print()

# ----------------------------------------------------------
# 3. Binary Evaluation (Supported vs Not-Supported)
# ----------------------------------------------------------
# We treat "Supported" as positive class for right answers
# and "Contradicted + Partially Supported" as positive for hallucinated answers

def map_to_binary(row):
    if row["answer_type"] == "right_answer":
        return 1 if row["verdict"] == "Supported" else 0
    else:  # hallucinated_answer
        return 1 if row["verdict"] in ["Contradicted", "Partially Supported"] else 0

df["binary_correct"] = df.apply(map_to_binary, axis=1)

y_true = [1 if x == "right_answer" else 0 for x in df["answer_type"]]  # 1 = should be supported
y_pred = df["binary_correct"].tolist()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("3. Binary Metrics (Treating correct detection)")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print()

# ----------------------------------------------------------
# 4. Abstention Rate
# ----------------------------------------------------------
abstention_rate = (df["verdict"] == "Unverifiable").mean() * 100
print(f"4. Abstention Rate (Unverifiable) : {abstention_rate:.1f}%")
print()

# ----------------------------------------------------------
# 5. Detailed Cross Table
# ----------------------------------------------------------
print("5. Detailed Counts (Answer Type vs Verdict)")
print(pd.crosstab(df["answer_type"], df["verdict"]))
print()

# ----------------------------------------------------------
# 6. Save a clean summary
# ----------------------------------------------------------
summary = {
    "total_claims": len(df),
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1": round(f1, 3),
    "abstention_rate": round(abstention_rate, 1),
    "verdict_distribution": df["verdict"].value_counts().to_dict()
}

with open("results/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Metrics summary saved to → results/metrics_summary1.json")
print("=" * 65)