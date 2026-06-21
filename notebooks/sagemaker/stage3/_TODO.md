# Stage 3 — SageMaker (belum ada notebook)

Belum ada versi SageMaker untuk Stage 3 (make_artifacts + GPT-4o/RAG).

Untuk sekarang pakai `../../colab/stage3/` (`02_make_artifacts.ipynb`, `04_explain_rag.ipynb`).
Port nanti: `/content/...` → `~/SageMaker/opg-data`, `drive.mount` → hapus, secret pakai file `~/SageMaker/.orkey`.
Script `scripts/stage3/*.py` dipakai apa adanya (parameter `--drive`).
