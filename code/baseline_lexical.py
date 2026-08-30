# code/baseline_lexical.py
import json, re
from pathlib import Path
from tqdm import tqdm

def overlap(claim, evidence):
    c = set(re.findall(r'\w+', claim.lower()))
    e = set(re.findall(r'\w+', evidence.lower()))
    return len(c & e) / len(c) if c else 0.0

with open("data/sample_300.json", encoding="utf-8") as f:
    data = json.load(f)

results = []
for idx, item in enumerate(tqdm(data)):
    knowledge = item["knowledge"]
    for ans_type in ["right_answer", "hallucinated_answer"]:
        claim = item[ans_type]
        o = overlap(claim, knowledge)
        if o >= 0.7:
            verdict = "Supported"
        elif o <= 0.15:
            verdict = "Contradicted"
        elif o >= 0.35:
            verdict = "Partially Supported"
        else:
            verdict = "Unverifiable"
        results.append({
            "id": idx,
            "answer_type": ans_type,
            "claim": claim,
            "verdict": verdict
        })

Path("results").mkdir(exist_ok=True)
with open("results/baseline_lexical.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Lexical baseline saved.")