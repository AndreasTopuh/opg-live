# OPG-Live

Vision-language explanation of panoramic dental radiographs with **SAM-based class-dependent grounding granularity**.
Thesis: *SAM-Based Class-Dependent Grounding Granularity in Vision-Language Explanation of Panoramic Dental Radiographs* — Andreas Jeno Figo Topuh (MSc AI, APU).

## Pipeline
```
OPG image ─▶ Stage 1: YOLOv8 (disease detector + enumeration → bbox + FDI + diagnosis)
          ─▶ Stage 2: SAM ViT-H + Medical SAM Adapter (lesion mask)
          ─▶ Stage 3: GPT-4o + RAG (L-F-V grounded explanation)
```
