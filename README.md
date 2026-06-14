# OPG-Live

Vision-language explanation of panoramic dental radiographs with **SAM-based class-dependent grounding granularity**.
Thesis: *SAM-Based Class-Dependent Grounding Granularity in Vision-Language Explanation of Panoramic Dental Radiographs* — Andreas Jeno Figo Topuh (MSc AI, APU).

## Pipeline
```
OPG image ─▶ Stage 1: HierarchicalDet (bbox + FDI + diagnosis, pretrained)
          ─▶ Stage 2: SAM ViT-H + Medical SAM Adapter (lesion mask)
          ─▶ Stage 3: GPT-4o + RAG (L-F-V grounded explanation)
```
3-arm comparison: **bounding box** / **lesion mask** / **hybrid** → faithfulness (HR, GS, CTC).

## Di mana dijalankan
| Lokasi | Tugas |
|---|---|
| 💻 **Laptop (VS Code)** | Edit `scripts/*.py`, struktur notebook, manual review KB chunks, GPT-4o call (API), validator, git |
| ☁️ **Colab Web (GPU)** | SAM ViT-H inference + adapter training, HierarchicalDet, BGE-M3 embed, full 3-arm batch |

> **Aturan:** GPU = Colab. Teks/API/edit = local. Checkpoint training WAJIB ke Google Drive.

## Struktur folder
```
OPG-Live/
├── scripts/         # semua *.py (edit di VS Code, run di Colab)
├── notebooks/       # 00_setup.ipynb dst (buka via "Open Colab Web")
├── backend/         # FastAPI (Phase C)
│   ├── routers/  services/  models/  templates/
├── frontend/        # React + Cornerstone.js (Phase C)
│   └── src/components  hooks  stores
├── data/            # GITIGNORED — di Colab = Google Drive
│   ├── dentex/      # DENTEX dataset (3,653 OPG, 3,529 polygon)
│   └── kb/          # RAG knowledge base (~30 chunk)
├── checkpoints/     # GITIGNORED — SAM adapter weights → Drive
├── outputs/         # GITIGNORED — masks / reports / metrics
└── docs/
```

## Quickstart (Colab)
1. `git push` dari laptop → buka `notebooks/00_setup.ipynb` via tombol **Open Colab Web**
2. Runtime → Change runtime type → **T4 GPU** (free) / L4 (Pro)
3. **Run All** → mount Drive, install deps, verify DENTEX, test SAM ViT-H

## Data & weights (TIDAK di GitHub)
Ada di Google Drive `MyDrive/opg-live/{data,checkpoints,outputs}`. Lihat `notebooks/00_setup.ipynb`.
