import json
import random
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import re

# ====================== CONFIG ======================
RESULTS_FILE = "results/stage3_results_v2.json"
OUTPUT_FILE = "results/faithfulness_test.json"
DEVICE = "cpu"
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# ====================================================

print("Loading model...")
model = CrossEncoder(MODEL_NAME, device=DEVICE)
print("Model loaded!")

def simple_overlap(claim: str, evidence: str) -> float:
    claim_tokens = set(re.findall(r'\w+', claim.lower()))
    evidence_tokens = set(re.findall(r'\w+', evidence.lower()))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)

def get_verdict(claim: str, evidence: str) -> str:
    scores = model.predict([(claim, evidence)])[0]
    contradiction, entailment, neutral = scores
    overlap = simple_overlap(claim, evidence)

    if entailment > 0.45 or (entailment > 0.30 and overlap > 0.6):
        return "Supported"
    if contradiction > 0.50:
        return "Contradicted"
    if entailment > 0.25 or (overlap > 0.4 and contradiction < 0.4):
        return "Partially Supported"
    return "Unverifiable"

def generate_explanation(claim: str, evidence: str, verdict: str) -> str:
    """Simple template-based explanation (faithful by construction)"""
    if verdict == "Supported":
        return f"The claim '{claim}' is supported by the evidence."
    elif verdict == "Contradicted":
        return f"The claim '{claim}' is contradicted by the evidence."
    elif verdict == "Partially Supported":
        return f"The claim '{claim}' is only partially supported by the evidence."
    else:
        return f"No sufficient evidence was found to verify the claim '{claim}'."

def run_faithfulness_test():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # We will test a sample of 100 claims to keep it fast on your laptop
    random.seed(42)
    sample = random.sample(results, 100)

    # Collect some real evidence texts to use as "swapped" evidence
    all_evidences = list(set([r["evidence_preview"] for r in results]))

    faithfulness_results = []
    changed_count = 0

    print("Running Counterfactual Evidence-Swap Test on 100 claims...")

    for item in tqdm(sample):
        claim = item["claim"]
        original_evidence = item["evidence_preview"]
        original_verdict = item["verdict"]

        # Generate original explanation
        original_explanation = generate_explanation(claim, original_evidence, original_verdict)

        # Swap with a random different evidence
        swapped_evidence = random.choice(all_evidences)
        while swapped_evidence == original_evidence:
            swapped_evidence = random.choice(all_evidences)

        # Get new verdict and explanation with swapped evidence
        new_verdict = get_verdict(claim, swapped_evidence)
        new_explanation = generate_explanation(claim, swapped_evidence, new_verdict)

        # Check if explanation changed
        explanation_changed = original_explanation != new_explanation
        if explanation_changed:
            changed_count += 1

        faithfulness_results.append({
            "claim": claim,
            "original_verdict": original_verdict,
            "new_verdict": new_verdict,
            "original_explanation": original_explanation,
            "new_explanation": new_explanation,
            "explanation_changed": explanation_changed
        })

    # Calculate faithfulness score
    faithfulness_score = (changed_count / len(sample)) * 100

    summary = {
        "total_tested": len(sample),
        "explanations_changed": changed_count,
        "faithfulness_score": round(faithfulness_score, 1),
        "details": faithfulness_results
    }

    Path("results").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("COUNTERFACTUAL FAITHFULNESS TEST RESULTS")
    print("="*60)
    print(f"Claims tested              : {len(sample)}")
    print(f"Explanations that changed  : {changed_count}")
    print(f"Faithfulness Score         : {faithfulness_score:.1f}%")
    print(f"Results saved to           : {OUTPUT_FILE}")
    print("="*60)
    print("\nInterpretation:")
    print("Higher score = better (explanation is sensitive to the evidence)")
    print("A score above 70-80% is considered good for this test.")

if __name__ == "__main__":
    run_faithfulness_test()