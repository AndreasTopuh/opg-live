# scripts/ — edit di VS Code, run di Colab

Semua logika pipeline sebagai `.py` (versi-controlled, di-`git push`). Notebook hanya orchestrate + jalankan di GPU.

| Script | Stage | Jalan di | Status | Fungsi |
|---|---|---|---|---|
| `dentex_dataset.py` | 2-prep | local/Colab | ✅ ada | DENTEX disease (hierarkis) → Dataset lesion (img+box+mask+cls), stratified split |
| `sam_adapter.py` | 2 | **Colab GPU** | ✅ ada | Medical SAM Adapter (Wu 2025): inject adapter, freeze base, ~2-3% trainable |
| `train_adapter.py` | 2 | **Colab GPU** | ✅ ada | Training loop box→mask, Dice+BCE, per-kelas Dice, checkpoint→Drive |
| `auto_pipeline.py` | 1+2 | **Colab GPU** | ⏳ next | upload → bbox+FDI+diagnosis → lesion mask |
| `extract_chunks.py` | 3-prep | local | ⏳ | pdfplumber: PDF guideline → ~30 chunk (lalu manual review) |
| `embed_kb.py` | 3-prep | **Colab GPU** | ⏳ | BGE-M3 embed chunks → `kb_embeddings.npy` |
| `retriever.py` | 3 | local | ⏳ | cosine similarity (NumPy) top-k chunk |
| `prompt_builder.py` | 3 | local | ⏳ | susun prompt 3-arm (bbox/mask/hybrid) + artifact |
| `llm_gpt.py` | 3 | local | ⏳ | GPT-4o via **OpenRouter** (`base_url`, model `openai/gpt-4o`) |
| `validator.py` | 3 | local | ⏳ | regex cek setiap sitasi output ada di KB |
| `ablation_grounding_granularity.py` | D | **Colab GPU** | ⏳ | 3-arm full batch → HR/GS/CTC |
| `analyze_user_study.py` | D | local | ⏳ | Friedman + Wilcoxon + Bonferroni + bootstrap CI |
