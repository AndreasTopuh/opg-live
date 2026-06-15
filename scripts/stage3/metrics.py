"""
Faithfulness metrics (draft §4.1): HR, GS, CTC. Computed automatically from the
structured L-F-V GPT-4o output + the KB.

(1) HR  Hallucination Rate = N_uncited / N findings.
        A finding is 'hallucinated' if it has no valid citation OR its content is
        insufficiently supported by the cited chunks (lexical support < thr).
(2) GS  Grounding Score = (findings with valid mask_id AND >=1 class-relevant citation) / N.
(3) CTC Cross-Tooth Contamination = fraction of EXPLANATIONS that reference a
        feature outside the target tooth. Automatic proxy: an FDI number != target
        in the text, or 'adjacent/neighbouring tooth' language. (NER+polygon = refinement.)
"""
import re

FDI_RE = re.compile(r"\b[1-4][1-8]\b")
ADJ_RE = re.compile(r"\b(adjacent|neighbou?ring|next tooth|surrounding teeth)\b", re.I)

# light stopwords so overlap focuses on clinical content words
STOP = set(
    "a an the of on in to for and or with is are be by that this it as at from "
    "your you their its within only both each per into over under can may will "
    "indicated tooth area suggests recommended confirm extent".split()
)


def _content_words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP]


def _support(value, chunk_texts):
    """Fraction of the finding's content words that appear in the cited chunks."""
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
    """A finding is 'hallucinated' if it has NO valid citation OR its content is
    insufficiently supported by the cited chunks (lexical support < thr). More
    realistic than verbatim trigram matching."""
    if not findings:
        return 1.0, 0, 0
    uncited = 0
    for f in findings:
        cites = [kb[c]["text"] for c in f.get("citations", []) if c in kb]
        if not cites or _support(finding_text(f), cites) < thr:
            uncited += 1
    return uncited / len(findings), uncited, len(findings)


def grounding_score(findings, kb, mask_id, disease=None):
    """A finding is 'grounded' if mask_id is valid AND it has >=1 class-relevant
    citation (chunk class == disease, or 'general'). If disease is None, fall back
    to checking valid citations only (backwards compatible)."""
    if not findings:
        return 0.0
    good = 0
    for f in findings:
        has_mask = bool(f.get("mask_id")) and f.get("mask_id") == mask_id
        cites = [c for c in f.get("citations", []) if c in kb]
        if disease is not None:
            cites = [c for c in cites if kb[c].get("class") in (disease, "general")]
        if has_mask and len(cites) >= 1:
            good += 1
    return good / len(findings)


def is_contaminated(findings, target_fdi=None):
    """True if the explanation references a tooth/feature outside the target."""
    text = " ".join(finding_text(f) + " " + str(f.get("location", "")) for f in findings)
    if target_fdi:
        fdis = set(FDI_RE.findall(text))
        fdis.discard(str(target_fdi))
        if fdis:
            return True
    return bool(ADJ_RE.search(text))


def explanation_metrics(findings, kb, mask_id, target_fdi=None, hr_thr=0.35, disease=None):
    hr, n_unc, n = hallucination_rate(findings, kb, hr_thr)
    gs = grounding_score(findings, kb, mask_id, disease)
    ctc = is_contaminated(findings, target_fdi)
    return {"HR": hr, "GS": gs, "CTC": int(ctc), "n_findings": n, "n_uncited": n_unc}
