# data/kb/ — RAG Knowledge Base (~30 chunk)

Manually-curated knowledge base untuk grounding penjelasan GPT-4o (Cara 2: manual KB, BGE-M3, NumPy cosine — no FAISS).

## Isi
- `chunks.jsonl` — ~30 chunk teks dari guideline (1 chunk = 1 fakta klinis + sumber)
- `kb_embeddings.npy` — BGE-M3 embeddings (GITIGNORED — generate via `embed_kb.py` di Colab)
- `sources/` — PDF guideline asli (AAE 2013, ICCMS/ICDAS 2014, DENTEX class definitions)

## Alur
1. `extract_chunks.py` (local) → draft chunks dari PDF
2. **Manual review** (L2) → koreksi, pastikan tiap chunk punya `source` + akurat klinis
3. `embed_kb.py` (Colab) → `kb_embeddings.npy`

## Format chunk
```json
{"id": "caries_def_01", "text": "...", "source": "ICCMS 2014, p.12", "class": "caries"}
```
