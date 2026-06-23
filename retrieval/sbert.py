# retrieval/sbert.py
import time
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.35

class SBERTSearch:
    def __init__(self, model_path, index_path, doc_path):
        self.embedder = SentenceTransformer(model_path)
        self.index = faiss.read_index(index_path)

        with open(doc_path, "rb") as f:
            self.documents = pickle.load(f)

    def embed(self, text):
        emb = self.embedder.encode([text], convert_to_numpy=True)
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    # ======================
    # 🔹 FOR EXPERIMENT
    # ======================
    def search(self, query, k=3):
        q_embedding = self.embed(query)
        scores, indices = self.index.search(q_embedding, k)
        return scores[0], indices[0]

    # ======================
    # 🔹 FOR APP (STREAMLIT)
    # ======================
    def search_with_details(self, query, k=3):
        start = time.time()

        q_embedding = self.embed(query)
        scores, indices = self.index.search(q_embedding, k)

        retrieval_time = time.time() - start

        results = []
        for rank, idx in enumerate(indices[0]):
            score = float(scores[0][rank])

            if score >= SIMILARITY_THRESHOLD:
                results.append({
                    "rank": rank + 1,
                    "text": self.documents[idx],
                    "score": score
                })

        if not results:
            return None, retrieval_time

        return results, retrieval_time