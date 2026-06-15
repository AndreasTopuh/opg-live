"""
RAG retriever: cosine similarity (NumPy) top-k chunks for a query.
KB is small (~28) -> brute-force cosine is already optimal, no FAISS needed.
"""
import json
import os

import numpy as np


class Retriever:
    def __init__(self, kb_dir):
        self.emb = np.load(f"{kb_dir}/kb_embeddings.npy")          # N x d (already normalized)
        self.chunks = json.load(open(f"{kb_dir}/kb_meta.json", encoding="utf-8"))
        self._model = None

    def _encode(self, text):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("BAAI/bge-m3")
        q = self._model.encode([text], normalize_embeddings=True)
        return np.asarray(q, dtype=np.float32)[0]

    def search(self, query, k=4, disease=None):
        """Top-k chunks. If disease is given, prioritise that class + general."""
        q = self._encode(query)
        sims = self.emb @ q                                        # cosine (normalized vectors)
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = dict(self.chunks[i])
            c["score"] = float(sims[i])
            out.append(c)
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    import sys

    kb = sys.argv[1] if len(sys.argv) > 1 else "/content/opg-live/data/kb"
    r = Retriever(kb)
    for c in r.search("radiolucency at the root apex non-vital tooth", k=3):
        print(round(c["score"], 3), c["id"], "-", c["text"][:70])
