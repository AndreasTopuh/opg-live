# scripts/ — edit di VS Code, run di Colab

Semua logika pipeline sebagai `.py` (versi-controlled, di-`git push`). Notebook hanya orchestrate + jalankan di GPU.

| Script | Stage | Jalan di | Status | Fungsi |
|---|---|---|---|---|
| `dentex_dataset.py` | 2-prep | local/Colab | ✅ ada | DENTEX disease (hierarkis) → Dataset lesion (img+box+mask+cls), stratified split |
| `sam_adapter.py` | 2 | **Colab GPU** | ✅ ada | Medical SAM Adapter (Wu 2025): inject adapter, freeze base, ~2-3% trainable |
| `train_adapter.py` | 2 | **Colab GPU** | ✅ ada | Training loop box→mask, Dice+BCE, per-kelas Dice, checkpoint→Drive. AMP fp16, decode per-gambar. Val Dice 0.94 |
| `dentex_to_yolo.py` | 1-prep | local/Colab | ✅ ada | DENTEX disease COCO → format YOLOv8 (split per-gambar) + dentex.yaml |
| `make_artifacts.py` | 1→2→3 | **Colab GPU** | ✅ ada | Deteksi YOLO Stage 1 (`--mode yolo`) → SAM+adapter mask → 3 arm + manifest (+ match GT untuk cek diagnosis benar). `--mode gt` = oracle sekunder |
| `auto_pipeline.py` | 1+2 | **Colab GPU** | ⏳ demo | YOLOv8 predict → mask (untuk demo deployment Phase C) |

> **Stage 1 = YOLOv8** (Plan B resmi roadmap; weights pretrained HierarchicalDet tidak dirilis publik). Selaras dengan supervisor Veerabhadrappa & Vengusamy (2025) yang memakai YOLOv7 di panoramic radiograph — di sini diperbarui ke YOLOv8. Lihat `notebooks/03_stage1_yolo.ipynb`.
| `extract_chunks.py` | 3-prep | local | ⏳ | pdfplumber: PDF guideline → ~30 chunk (lalu manual review) |
| `embed_kb.py` | 3-prep | Colab | ✅ ada | BGE-M3 embed ~28 chunk → `kb_embeddings.npy` + `kb_meta.json` |
| `retriever.py` | 3 | local/Colab | ✅ ada | cosine similarity (NumPy) top-k chunk, no FAISS |
| `prompt_builder.py` | 3 | local | ✅ ada | prompt 3-arm (referent bbox/mask/hybrid) + skema L-F-V JSON |
| `llm_gpt.py` | 3 | local | ✅ ada | GPT-4o vision via **OpenRouter** (`base_url`, `openai/gpt-4o`), JSON out |
| `metrics.py` | 3 | local | ✅ ada | HR (trigram-cite), GS (mask_id+citation), CTC (FDI/adjacency proxy) |
| `run_stage3.py` | 3 | local/Colab | ✅ ada | loop deteksi×3 arm → RAG → GPT-4o → metrik → `results.jsonl` (`--limit` utk tes) |
| `analyze_results.py` | D | local | ✅ ada | mean+bootstrap CI per arm, Friedman+Wilcoxon+Bonferroni, per-kelas H2-H5 |
