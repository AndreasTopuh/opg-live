"""
Build the 3-arm prompt + L-F-V (Location-Field-Value) schema for GPT-4o.

The 3 arms differ ONLY in the spatial-referent description (single variable, §4.3):
  bbox   : green box
  mask   : red mask overlay
  hybrid : both
The diagnosis (class), FDI, and RAG chunks are IDENTICAL across arms (held constant).

GPT-4o output is forced to L-F-V JSON with mask_id + citations (for GS/HR metrics).
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
    return f"""You are a dental radiology decision-support assistant explaining a finding on a panoramic radiograph (OPG). Base your explanation on what the indicated region shows and on what the knowledge base supports.

NOTE ON NUMBERING: any two-digit tooth identifier (e.g. 11, 36, 46) is FDI World Dental Federation notation (quadrant + tooth position), NOT a statistic or measurement.

SPATIAL GROUNDING: {ARM_DESC[arm]}
DETECTOR OUTPUT (Stage 1): condition = "{disease}" on {fdi_txt}.

KNOWLEDGE BASE (cite findings by these ids only):
{kb}

GUIDELINES:
1. Explain the finding indicated by the spatial referent, grounded in the visible region and the knowledge base.
2. Every finding MUST set "mask_id" to "{mask_id}" and include at least one "citations" id from: {cite_ids}.
3. Be concise and clinically accurate; lower the confidence if uncertain.

Return ONLY valid JSON (no prose) in this schema:
{{"findings": [
  {{"location": "{fdi or 'indicated tooth'}", "field": "finding", "value": "<one concise clinical sentence>", "confidence": 0.0, "mask_id": "{mask_id}", "evidence": ["<visible radiographic feature>"], "citations": ["<chunk_id>"]}},
  {{"location": "{fdi or 'indicated tooth'}", "field": "recommendation", "value": "<one concise management sentence>", "confidence": 0.0, "mask_id": "{mask_id}", "evidence": [], "citations": ["<chunk_id>"]}}
]}}"""
