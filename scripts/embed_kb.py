"""
Embed the knowledge base (~28 manual chunks) with BGE-M3 -> kb_embeddings.npy.

RAG "approach 2": manual KB, BGE-M3 embeddings, NumPy cosine (no FAISS, no
auto-chunking). Run once (on Colab GPU, fast). Output is used by retriever.py.
"""
import argparse
import json
import os

import numpy as np


def run(args):
    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    print(f"Chunks: {len(chunks)}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    out_dir = os.path.dirname(args.chunks)
    np.save(f"{out_dir}/kb_embeddings.npy", emb)
    with open(f"{out_dir}/kb_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"OK: embeddings {emb.shape} -> {out_dir}/kb_embeddings.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="/content/opg-live/data/kb/chunks.jsonl")
    run(ap.parse_args())
