import numpy as np


# ======================
# BASIC METRICS
# ======================

def precision_at_k(ranked, relevant, k):
    ranked_k = ranked[:k]
    hits = sum(1 for doc in ranked_k if doc in relevant)
    return hits / k


def recall_at_k(ranked, relevant, k):
    ranked_k = ranked[:k]
    hits = sum(1 for doc in ranked_k if doc in relevant)
    return hits / len(relevant) if relevant else 0


def reciprocal_rank(ranked, relevant):
    for i, doc in enumerate(ranked):
        if doc in relevant:
            return 1 / (i + 1)
    return 0


def hit_at_k(ranked, relevant, k):
    return int(any(doc in relevant for doc in ranked[:k]))


# ======================
# DCG & nDCG
# ======================

def dcg_at_k(ranked, relevant, k):
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            dcg += 1 / np.log2(i + 2)
    return dcg


def ndcg_at_k(ranked, relevant, k):
    # Ideal DCG (ranking sempurna)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg_at_k(ranked, relevant, k) / idcg


# ======================
# MAIN EVALUATOR
# ======================

def evaluate(search_fn, queries, relevance, k=3, verbose=False):
    precisions = []
    recalls = []
    mrrs = []
    hits = []
    ndcgs = []

    for i, query in enumerate(queries):
        # ======================
        # GET RESULT
        # ======================
        result = search_fn(query, k)

        # Support dua format:
        # (scores, ranked) atau hanya ranked
        if isinstance(result, tuple):
            scores, ranked = result
        else:
            ranked = result
            scores = None

        # ======================
        # SAFETY CHECK
        # ======================
        if ranked is None:
            raise ValueError(f"Ranked result is None for query: {query}")

        ranked = list(ranked)
        rel_docs = relevance.get(i, [])

        # ======================
        # DEBUG LOG
        # ======================
        if verbose:
            print("\n======================")
            print(f"Query: {query}")
            print(f"Top-{k}: {ranked}")
            print(f"Relevant: {rel_docs}")

        # ======================
        # METRICS
        # ======================
        precisions.append(precision_at_k(ranked, rel_docs, k))
        recalls.append(recall_at_k(ranked, rel_docs, k))
        mrrs.append(reciprocal_rank(ranked, rel_docs))
        hits.append(hit_at_k(ranked, rel_docs, k))
        ndcgs.append(ndcg_at_k(ranked, rel_docs, k))

    # ======================
    # FINAL RESULT
    # ======================
    return {
        "Precision@k": float(np.mean(precisions)),
        "Recall@k": float(np.mean(recalls)),
        "MRR": float(np.mean(mrrs)),
        "Hit@k": float(np.mean(hits)),
        "nDCG@k": float(np.mean(ndcgs))
    }