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
| `embed_kb.py` | 3-prep | **Colab GPU** | ⏳ | BGE-M3 embed chunks → `kb_embeddings.npy` |
| `retriever.py` | 3 | local | ⏳ | cosine similarity (NumPy) top-k chunk |
| `prompt_builder.py` | 3 | local | ⏳ | susun prompt 3-arm (bbox/mask/hybrid) + artifact |
| `llm_gpt.py` | 3 | local | ⏳ | GPT-4o via **OpenRouter** (`base_url`, model `openai/gpt-4o`) |
| `validator.py` | 3 | local | ⏳ | regex cek setiap sitasi output ada di KB |
| `ablation_grounding_granularity.py` | D | **Colab GPU** | ⏳ | 3-arm full batch → HR/GS/CTC |
| `analyze_user_study.py` | D | local | ⏳ | Friedman + Wilcoxon + Bonferroni + bootstrap CI |
