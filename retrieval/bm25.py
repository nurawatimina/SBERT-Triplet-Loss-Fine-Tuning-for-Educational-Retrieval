from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, k=3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [scores[i] for i in ranked_idx], ranked_idx