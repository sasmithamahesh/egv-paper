from sentence_transformers import CrossEncoder
import time

print("Loading small NLI model on CPU...")
model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device='cpu')

claim = "Marie Curie was born in Warsaw."
evidence = "Marie Curie was born in Warsaw, Poland in 1867."

start = time.time()
scores = model.predict([(claim, evidence)])
print("Score:", scores)
print("Time taken:", round(time.time() - start, 2), "seconds")