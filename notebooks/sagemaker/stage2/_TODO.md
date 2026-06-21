# Stage 2 — SageMaker (belum ada notebook terpisah)

Training SAM adapter Stage 2 untuk SageMaker **sudah ada di `../00_setup_sagemaker.ipynb` Cell 9**
(`train_adapter.py --drive ~/SageMaker/opg-data --bs 1 --accum 8`).

Kalau mau notebook Stage 2 SageMaker yang berdiri sendiri: port `../../colab/stage2/01_sam_adapter_train.ipynb`
→ ganti `drive.mount` + `/content/...` jadi path EBS `~/SageMaker/opg-data` (script `train_adapter.py` sendiri tidak berubah, cuma `--drive`).
