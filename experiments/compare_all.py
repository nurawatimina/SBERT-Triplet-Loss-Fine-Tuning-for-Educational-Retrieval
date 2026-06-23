import pandas as pd
import os
import matplotlib.pyplot as plt

from retrieval.sbert import SBERTSearch
from evaluation.evaluator import evaluate

# ======================
# CONFIG
# ======================
K = 10
RESULT_PATH = "results/final_comparison.csv"

# ======================
# LOAD DATA
# ======================
print("Loading dataset...")
df = pd.read_csv("data/qa_dataset.csv")

documents = df["answer"].tolist()
queries = df["question"].tolist()

relevance = {i: [i] for i in range(len(queries))}

# ======================
# MODELS
# ======================
models = {
    "SBERT Baseline": "all-MiniLM-L6-v2",
    "Triplet Hard": "model/triplet_hard",
    "Triplet Semi": "model/triplet_semi",
    "Triplet Random": "model/triplet_random",
    "Triplet Mixed": "model/triplet_all",
}

# ======================
# RUN
# ======================
results = []

print("\nRunning comparison...\n")

for name, model_path in models.items():
    print(f"=== {name} ===")

    search = SBERTSearch(
        model_path=model_path,
        index_path="index/baseline/faiss.index",
        doc_path="index/baseline/documents.pkl"
    )

    metrics = evaluate(
        search_fn=search.search,
        queries=queries,
        relevance=relevance,
        k=K
    )

    metrics["Model"] = name
    results.append(metrics)

    print(metrics)

# ======================
# SAVE
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
metrics_list = ["MRR", "nDCG@k", "Precision@k"]

for metric in metrics_list:
    plt.figure()
    plt.bar(df_results["Model"], df_results[metric])
    plt.title(metric)
    plt.xticks(rotation=20)
    plt.savefig(f"results/{metric}_comparison.png")
    plt.close()

print("\nDone.")