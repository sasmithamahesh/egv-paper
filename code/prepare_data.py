import json
import random
from pathlib import Path

print("Loading HaluEval QA data (JSON Lines format)...")

data = []
with open("data/qa_data.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            data.append(json.loads(line))

print(f"Total items loaded: {len(data)}")

# Create a reproducible sample of 300 items
random.seed(42)
sample = random.sample(data, 300)

# Save the sample
Path("data").mkdir(exist_ok=True)
with open("data/sample_300.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)

print("Successfully saved 300-item sample → data/sample_300.json")
print("You can now run the next step.")