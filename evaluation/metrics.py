def precision_at_k(ranked, relevant, k):
    return sum([1 for d in ranked[:k] if d in relevant]) / k

def recall_at_k(ranked, relevant, k):
    return sum([1 for d in ranked[:k] if d in relevant]) / len(relevant)

def reciprocal_rank(ranked, relevant):
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1 / (i + 1)
    return 0