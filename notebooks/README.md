# Notebooks — per platform, per stage

Notebook = orkestrator tipis; semua logika ada di `../scripts/{stage1,stage2,stage3}/`
(dipanggil dengan parameter `--drive <root>`). Struktur notebook **mirror `scripts/`**:
dibagi per platform (Colab vs SageMaker) lalu per stage.

```
notebooks/
├── colab/                     # run di Google Colab (drive.mount, /content)
│   ├── 00_setup.ipynb         # setup (mount Drive, deps, SAM ViT-H, verify DENTEX)
│   ├── 06_mvp_demo.ipynb      # demo end-to-end
│   ├── stage1/                # 03_stage1_yolo (disease) · 05_stage1_enum (FDI)
│   ├── stage2/                # 01_sam_adapter_train (SAM adapter)
│   └── stage3/                # 02_make_artifacts (3-arm) · 04_explain_rag (GPT-4o+RAG)
└── sagemaker/                 # run di AWS SageMaker (EBS ~/SageMaker/opg-data)
    ├── 00_setup_sagemaker.ipynb   # setup + upload DENTEX + (Cell 9) Stage 2 train
    ├── stage1/                # 03_stage1_yolo_sagemaker (A10G 24GB, batch 16)
    ├── stage2/                # _TODO (pakai 00_setup_sagemaker Cell 9 / port colab)
    └── stage3/                # _TODO (port dari colab/stage3)
```

## Colab vs SageMaker — bedanya cuma environment

| | Colab | SageMaker |
|---|---|---|
| Akses data | `drive.mount()` → Google Drive | EBS `~/SageMaker/opg-data/` |
| `--drive` root | `/content/drive/MyDrive/opg-live` | `~/SageMaker/opg-data` |
| GPU | T4 / L4 / A100 | A10G 24GB (g5.xlarge) atau T4 (g4dn.xlarge) |
| Secret LLM | `google.colab.userdata` | file `~/SageMaker/.orkey` |
| Backup | Google Drive | S3 (`aws s3 sync`) |

## Cara port Colab → SageMaker (kalau bikin versi SageMaker baru)
1. Buang `from google.colab import drive` + `drive.mount(...)`.
2. `/content/drive/MyDrive/opg-live` → `os.path.expanduser('~/SageMaker/opg-data')`.
3. `/content/...` scratch (mis. `/content/yolo`) → `~/SageMaker/...`.
4. Secret: `userdata.get(...)` → baca file `~/SageMaker/.orkey`.
5. **Script `../scripts/*.py` TIDAK berubah** — hanya nilai `--drive` yang beda.

## Catatan
Data DENTEX di-upload **sekali per platform**: Colab → Drive; SageMaker → `~/SageMaker/opg-data/data/dentex/`
(lewat `sagemaker/00_setup_sagemaker.ipynb` Cell 4 / gdown).
