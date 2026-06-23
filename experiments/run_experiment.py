import os
import pandas as pd
import matplotlib.pyplot as plt

from retrieval.tfidf import TFIDFSearch
from retrieval.bm25 import BM25Search
from retrieval.sbert import SBERTSearch
from evaluation.evaluator import evaluate

# ======================
# CONFIG
# ======================
K = 10
VERBOSE = False
RESULT_PATH = "results/experiment_results.csv"

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
print("\nInitializing models...")

tfidf = TFIDFSearch(documents)
bm25 = BM25Search(documents)

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
    "TF-IDF": tfidf.search,
    "BM25": bm25.search,
    "SBERT Baseline": sbert_baseline.search,
    "SBERT Triplet": sbert_triplet.search
}

# ======================
# RUN EXPERIMENT
# ======================
print("\nRunning experiments...\n")

results = []

for name, method in methods.items():
    print(f"=== {name} ===")

    try:
        metrics = evaluate(
            search_fn=method,
            queries=queries,
            relevance=relevance,
            k=K,
            verbose=VERBOSE
        )

        metrics["Model"] = name
        results.append(metrics)

        print(metrics)

    except Exception as e:
        print(f"Error in {name}: {e}")

# ======================
# SAVE RESULTS
# ======================
print("\nSaving results...")

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
print("\nGenerating visualization...")

metrics_list = ["Precision@k", "Recall@k", "MRR", "Hit@k", "nDCG@k"]

for metric in metrics_list:
    plt.figure()

    plt.bar(df_results["Model"], df_results[metric])

    plt.title(metric)
    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.xticks(rotation=20)

    save_path = f"results/{metric}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")

print("\nDone.")