"""
Recompute HR/GS/CTC from the saved results.jsonl 'raw' GPT-4o responses.
NO GPT-4o calls -> FREE. Use this to tune metric thresholds/formulas without
re-spending on the API. Only prompt/model changes require a paid re-run.
"""
import argparse
import json

from metrics import explanation_metrics, kb_by_id


def run(args):
    kb = kb_by_id(json.load(open(f"{args.kb}/kb_meta.json", encoding="utf-8")))
    path = f"{args.drive}/outputs/metrics/results.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    out = []
    for r in rows:
        try:
            findings = (json.loads(r["raw"]) or {}).get("findings", [])
        except (json.JSONDecodeError, TypeError):
            findings = []
        met = explanation_metrics(
            findings, kb, r["det_id"],
            target_fdi=r.get("target_fdi"),
            hr_thr=args.hr_thr,
            disease=r["pred_cls_name"],
        )
        out.append({**r, **met})

    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"Recomputed {len(out)} rows from saved raw responses (no API calls).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--kb", default="/content/opg-live/data/kb")
    ap.add_argument("--hr_thr", type=float, default=0.35)
    run(ap.parse_args())
