"""
Metrik faithfulness (§4.1 draft): HR, GS, CTC. Dihitung otomatis dari output
L-F-V terstruktur GPT-4o + KB.

(1) HR  Hallucination Rate = N_uncited / N findings.
        Sebuah finding lolos jika punya >=1 citation yang teks chunk-nya berbagi
        >= thr trigram (char) dgn 'value' finding; gagal -> hallucinated.
(2) GS  Grounding Score = (findings dgn mask_id valid AND >=1 citation valid) / N.
(3) CTC Cross-Tooth Contamination = proporsi EXPLANATION yang menyebut fitur di
        luar gigi target. Proxy otomatis: ada nomor FDI != target di teks, atau
        bahasa 'adjacent/neighbouring tooth'. (Versi NER+poligon = refinement.)
"""
import re

FDI_RE = re.compile(r"\b[1-4][1-8]\b")
ADJ_RE = re.compile(r"\b(adjacent|neighbou?ring|next tooth|surrounding teeth)\b", re.I)

# stopword ringan supaya overlap fokus ke kata-isi klinis
STOP = set(
    "a an the of on in to for and or with is are be by that this it as at from "
    "your you their its within only both each per into over under can may will "
    "indicated tooth area suggests recommended confirm extent".split()
)


def _content_words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP]


def _support(value, chunk_texts):
    """Fraksi kata-isi finding yang muncul di gabungan chunk tersitasi."""
    vw = _content_words(value)
    if not vw:
        return 1.0
    cw = set()
    for t in chunk_texts:
        cw |= set(_content_words(t))
    return sum(1 for w in vw if w in cw) / len(vw)


def kb_by_id(chunks):
    return {c["id"]: c for c in chunks}


def finding_text(f):
    return " ".join([str(f.get("value", ""))] + [str(e) for e in f.get("evidence", [])])


def hallucination_rate(findings, kb, thr=0.35):
    """Finding 'halusinasi' jika TIDAK punya citation valid ATAU isi-nya kurang
    didukung chunk tersitasi (lexical support < thr). Lebih realistis daripada
    pencocokan trigram verbatim."""
    if not findings:
        return 1.0, 0, 0
    uncited = 0
    for f in findings:
        cites = [kb[c]["text"] for c in f.get("citations", []) if c in kb]
        if not cites or _support(finding_text(f), cites) < thr:
            uncited += 1
    return uncited / len(findings), uncited, len(findings)


def grounding_score(findings, kb, mask_id):
    if not findings:
        return 0.0
    good = 0
    for f in findings:
        has_mask = bool(f.get("mask_id")) and f.get("mask_id") == mask_id
        valid_cites = [c for c in f.get("citations", []) if c in kb]
        if has_mask and len(valid_cites) >= 1:
            good += 1
    return good / len(findings)


def is_contaminated(findings, target_fdi=None):
    """True jika explanation menyebut gigi/fitur di luar target."""
    text = " ".join(finding_text(f) + " " + str(f.get("location", "")) for f in findings)
    if target_fdi:
        fdis = set(FDI_RE.findall(text))
        fdis.discard(str(target_fdi))
        if fdis:
            return True
    return bool(ADJ_RE.search(text))


def explanation_metrics(findings, kb, mask_id, target_fdi=None, hr_thr=0.8):
    hr, n_unc, n = hallucination_rate(findings, kb, hr_thr)
    gs = grounding_score(findings, kb, mask_id)
    ctc = is_contaminated(findings, target_fdi)
    return {"HR": hr, "GS": gs, "CTC": int(ctc), "n_findings": n, "n_uncited": n_unc}
