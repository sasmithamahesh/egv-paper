import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ====================== LOAD NEW RESULTS ======================
with open("results/stage3_results_v2.json", "r", encoding="utf-8") as f:
    results = json.load(f)

df = pd.DataFrame(results)

print("=" * 65)
print("EGV Stage 3 v2 - Improved Evaluation")
print("=" * 65)

print("\n1. Verdict Distribution (out of 600 claims)")
print(df["verdict"].value_counts())
print()

right = df[df["answer_type"] == "right_answer"]
hallu = df[df["answer_type"] == "hallucinated_answer"]

print("2. Performance Breakdown")
print(f"Right Answers marked Supported        : {(right['verdict'] == 'Supported').sum()} / 300  ({(right['verdict'] == 'Supported').mean()*100:.1f}%)")
print(f"Hallucinated Answers correctly caught : {((hallu['verdict'] == 'Contradicted') | (hallu['verdict'] == 'Partially Supported')).sum()} / 300  ({((hallu['verdict'] == 'Contradicted') | (hallu['verdict'] == 'Partially Supported')).mean()*100:.1f}%)")
print()

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

print("3. Binary Metrics")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print()

abstention_rate = (df["verdict"] == "Unverifiable").mean() * 100
print(f"4. Abstention Rate (Unverifiable) : {abstention_rate:.1f}%")
print()

print("5. Detailed Counts")
print(pd.crosstab(df["answer_type"], df["verdict"]))
print("=" * 65)