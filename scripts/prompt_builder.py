"""
Bangun prompt 3-arm + skema L-F-V (Location-Field-Value) untuk GPT-4o.

3 arm beda HANYA pada deskripsi spatial referent (single-variable, §4.3):
  bbox   : kotak hijau
  mask   : overlay mask merah
  hybrid : keduanya
Diagnosis (kelas), FDI, dan chunk RAG IDENTIK lintas arm (held constant).

Output GPT-4o dipaksa JSON L-F-V dengan mask_id + citations (untuk metrik GS/HR).
"""

ARM_DESC = {
    "bbox": "The lesion is indicated ONLY by the GREEN BOUNDING BOX drawn on the radiograph. Treat the box as the spatial referent.",
    "mask": "The lesion is indicated ONLY by the RED SEGMENTATION MASK overlay on the radiograph. Treat the masked region as the spatial referent.",
    "hybrid": "The lesion is indicated by BOTH the GREEN BOUNDING BOX and the RED SEGMENTATION MASK overlay. Use both as the spatial referent.",
}


def build_prompt(arm, disease, fdi, chunks, mask_id):
    kb = "\n".join(f"[{c['id']}] {c['text']} (Source: {c['source']})" for c in chunks)
    fdi_txt = f"tooth FDI {fdi}" if fdi else "the indicated tooth"
    cite_ids = ", ".join(c["id"] for c in chunks)
    return f"""You are a dental radiology decision-support assistant explaining ONE finding on a panoramic radiograph (OPG). Your explanation must be faithful: state only what is supported by the indicated region and the knowledge base.

SPATIAL GROUNDING: {ARM_DESC[arm]}
DETECTOR OUTPUT (Stage 1): condition = "{disease}" on {fdi_txt}.

KNOWLEDGE BASE (cite findings by these ids only):
{kb}

RULES:
1. Describe ONLY the lesion within the indicated spatial referent. Do NOT describe other teeth, other regions, or features outside the indicated lesion.
2. Every finding MUST set "mask_id" to "{mask_id}" and include at least one "citations" id from: {cite_ids}.
3. Be concise and clinically accurate. If unsure, lower the confidence.

Return ONLY valid JSON (no prose) in this schema:
{{"findings": [
  {{"location": "{fdi or 'indicated tooth'}", "field": "finding", "value": "<one concise clinical sentence>", "confidence": 0.0, "mask_id": "{mask_id}", "evidence": ["<visible radiographic feature>"], "citations": ["<chunk_id>"]}},
  {{"location": "{fdi or 'indicated tooth'}", "field": "recommendation", "value": "<one concise management sentence>", "confidence": 0.0, "mask_id": "{mask_id}", "evidence": [], "citations": ["<chunk_id>"]}}
]}}"""
