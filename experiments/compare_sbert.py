import os
import pandas as pd
import matplotlib.pyplot as plt

from retrieval.sbert import SBERTSearch
from evaluation.evaluator import evaluate

# ======================
# CONFIG
# ======================
K = 10
RESULT_PATH = "results/sbert_comparison.csv"

# ======================
# LOAD DATA
# ======================
print("Loading dataset...")
df = pd.read_csv("data/qa_dataset.csv")

documents = df["answer"].tolist()
queries = df["question"].tolist()

print(f"Total queries: {len(queries)}")

# ======================
# GROUND TRUTH
# ======================
relevance = {i: [i] for i in range(len(queries))}

# ======================
# INIT MODELS
# ======================
print("\nLoading SBERT models...")

sbert_baseline = SBERTSearch(
    model_path="all-MiniLM-L6-v2",
    index_path="index/baseline/faiss.index",
    doc_path="index/baseline/documents.pkl"
)

sbert_triplet = SBERTSearch(
    model_path="model/triplet-hard",
    index_path="index/hard/faiss.index",
    doc_path="index/hard/documents.pkl"
)

# ======================
# METHODS
# ======================
methods = {
    "SBERT Baseline": sbert_baseline.search,
    "SBERT Triplet": sbert_triplet.search
}

# ======================
# RUN EXPERIMENT
# ======================
print("\nRunning SBERT comparison...\n")

results = []

for name, method in methods.items():
    print(f"=== {name} ===")

    metrics = evaluate(
        search_fn=method,
        queries=queries,
        relevance=relevance,
        k=K,
        verbose=False
    )

    metrics["Model"] = name
    results.append(metrics)

    print(metrics)

# ======================
# SAVE RESULTS
# ======================
os.makedirs("results", exist_ok=True)

df_results = pd.DataFrame(results)
df_results = df_results[
    ["Model", "Precision@k", "Recall@k", "MRR", "Hit@k", "nDCG@k"]
]

print("\nFinal Results:")
print(df_results)

df_results.to_csv(RESULT_PATH, index=False)
print(f"\nSaved to {RESULT_PATH}")

# ======================
# VISUALIZATION
# ======================
print("\nGenerating comparison chart...")

metrics_list = ["Precision@k", "MRR", "nDCG@k"]

for metric in metrics_list:
    plt.figure()

    plt.bar(df_results["Model"], df_results[metric])

    plt.title(f"SBERT Comparison - {metric}")
    plt.ylabel(metric)

    save_path = f"results/sbert_{metric}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")

print("\nDone.")