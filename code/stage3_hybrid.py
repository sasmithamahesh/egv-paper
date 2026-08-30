import json
import re
import time
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# ====================== CONFIG ======================
SAMPLE_FILE = "data/sample_300.json"
OUTPUT_FILE = "results/stage3_hybrid.json"
DEVICE = "cpu"
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# ====================================================

print("Loading model on CPU...")
model = CrossEncoder(MODEL_NAME, device=DEVICE)
print("Model loaded!")

def token_overlap(claim: str, evidence: str) -> float:
    claim_toks = set(re.findall(r'\w+', claim.lower()))
    evid_toks = set(re.findall(r'\w+', evidence.lower()))
    if not claim_toks:
        return 0.0
    return len(claim_toks & evid_toks) / len(claim_toks)

def get_verdict(claim: str, evidence: str) -> str:
    scores = model.predict([(claim, evidence)])[0]  # [contradiction, entailment, neutral]
    contradiction, entailment, neutral = scores
    overlap = token_overlap(claim, evidence)
    claim_len = len(claim.split())

    # --- Improved hybrid rules ---
    # Short answers (very common in HaluEval)
    if claim_len <= 4:
        if overlap >= 0.75:
            return "Supported"
        if contradiction > 0.55:
            return "Contradicted"
        if overlap >= 0.40:
            return "Partially Supported"
        return "Unverifiable"

    # Longer answers
    if entailment > 0.50 or (entailment > 0.35 and overlap > 0.55):
        return "Supported"
    if contradiction > 0.55:
        return "Contradicted"
    if entailment > 0.28 or overlap > 0.45:
        return "Partially Supported"
    return "Unverifiable"

def main():
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    start = time.time()

    for idx, item in enumerate(tqdm(data, desc="Hybrid Stage 3")):
        knowledge = item["knowledge"]
        for ans_type in ["right_answer", "hallucinated_answer"]:
            claim = item[ans_type]
            verdict = get_verdict(claim, knowledge)
            results.append({
                "id": idx,
                "answer_type": ans_type,
                "claim": claim,
                "verdict": verdict,
                "gold_is_hallucinated": ans_type == "hallucinated_answer"
            })

        if (idx + 1) % 20 == 0:
            Path("results").mkdir(exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    Path("results").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {(time.time()-start)/60:.1f} minutes")
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()