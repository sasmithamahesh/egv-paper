import json
import time
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import numpy as np

# ====================== CONFIG ======================
SAMPLE_FILE = "data/sample_300.json"
OUTPUT_FILE = "results/stage3_results_v2.json"
DEVICE = "cpu"
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# ====================================================

print("Loading NLI model on CPU...")
model = CrossEncoder(MODEL_NAME, device=DEVICE)
print("Model loaded!")

def simple_overlap(claim: str, evidence: str) -> float:
    """Simple token overlap as extra signal for short answers"""
    claim_tokens = set(re.findall(r'\w+', claim.lower()))
    evidence_tokens = set(re.findall(r'\w+', evidence.lower()))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)

def get_four_way_verdict(claim: str, evidence: str) -> str:
    """
    Improved mapping for short answers in HaluEval.
    Combines NLI scores + simple lexical overlap.
    """
    scores = model.predict([(claim, evidence)])[0]  # [contradiction, entailment, neutral]
    contradiction, entailment, neutral = scores
    overlap = simple_overlap(claim, evidence)

    # --- Improved decision rules ---
    # Strong support
    if entailment > 0.45 or (entailment > 0.30 and overlap > 0.6):
        return "Supported"
    
    # Strong contradiction
    if contradiction > 0.50:
        return "Contradicted"
    
    # Partial support
    if entailment > 0.25 or (overlap > 0.4 and contradiction < 0.4):
        return "Partially Supported"
    
    # Everything else
    return "Unverifiable"

def process_sample():
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    start_time = time.time()

    for idx, item in enumerate(tqdm(data, desc="Processing claims")):
        knowledge = item["knowledge"]

        for answer_type in ["right_answer", "hallucinated_answer"]:
            claim = item[answer_type]

            verdict = get_four_way_verdict(claim, knowledge)

            results.append({
                "id": idx,
                "answer_type": answer_type,
                "claim": claim,
                "evidence_preview": knowledge[:300] + "..." if len(knowledge) > 300 else knowledge,
                "verdict": verdict,
                "gold_is_hallucinated": answer_type == "hallucinated_answer"
            })

        # Save progress every 20 items
        if (idx + 1) % 20 == 0:
            Path("results").mkdir(exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    Path("results").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print(f"\nFinished! Time taken: {total_time/60:.1f} minutes")
    print(f"Results saved to → {OUTPUT_FILE}")
    return results

if __name__ == "__main__":
    process_sample()