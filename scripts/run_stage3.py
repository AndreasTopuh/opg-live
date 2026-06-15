"""
Stage 3 orchestrator: for each detection × 3 arms -> RAG retrieve -> GPT-4o (OpenRouter)
-> parse L-F-V -> HR/GS/CTC metrics. Save results.jsonl to Drive.

RAG chunks are RETRIEVED ONCE per detection and reused across the 3 arms (held
constant, §4.3). Only the spatial referent (arm) changes.

Use --limit for a cheap test first (e.g. --limit 3 = 9 GPT-4o calls) before the full run.
"""
import argparse
import json
import os
import time

from llm_gpt import explain, get_client
from metrics import explanation_metrics, kb_by_id
from prompt_builder import build_prompt
from retriever import Retriever

ARMS = ["bbox", "mask", "hybrid"]


def run(args):
    art_dir = f"{args.drive}/outputs/artifacts"
    manifest = [json.loads(l) for l in open(f"{art_dir}/manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"Detections: {len(manifest)} × {len(ARMS)} arms = {len(manifest)*len(ARMS)} GPT-4o calls")

    retr = Retriever(args.kb)
    kb = kb_by_id(retr.chunks)
    client = get_client()

    out_path = f"{args.drive}/outputs/metrics/results.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")

    n_done, n_fail = 0, 0
    for mi, m in enumerate(manifest):
        disease = m["pred_cls_name"]
        fdi = m.get("target_fdi")
        mask_id = m["det_id"]
        # RAG once per detection (same for all 3 arms)
        query = f"{disease} on panoramic dental radiograph: radiographic appearance and management"
        chunks = retr.search(query, k=args.k)

        for arm in ARMS:
            img = f"{art_dir}/{m['artifacts'][arm]}"
            prompt = build_prompt(arm, disease, fdi, chunks, mask_id)
            try:
                parsed, raw = explain(client, prompt, img, model=args.model)
            except Exception as e:
                print(f"  [{m['det_id']}|{arm}] API error: {e}")
                n_fail += 1
                time.sleep(2)
                continue

            findings = (parsed or {}).get("findings", [])
            met = explanation_metrics(findings, kb, mask_id, target_fdi=fdi,
                                      hr_thr=args.hr_thr, disease=disease)
            fout.write(json.dumps({
                "det_id": m["det_id"],
                "arm": arm,
                "pred_cls": m["pred_cls"],
                "pred_cls_name": disease,
                "target_fdi": fdi,
                "correct_dx": m.get("correct"),
                "retrieved": [c["id"] for c in chunks],
                "parsed_ok": parsed is not None,
                **met,
                "raw": raw,
            }) + "\n")
            fout.flush()
            n_done += 1
        if (mi + 1) % 10 == 0:
            print(f"  {mi+1}/{len(manifest)} detections done")

    fout.close()
    print(f"\nOK: {n_done} results ({n_fail} failed) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--kb", default="/content/opg-live/data/kb")
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=4)            # top-k RAG chunks
    ap.add_argument("--hr_thr", type=float, default=0.35)  # lexical support threshold
    ap.add_argument("--limit", type=int, default=0)        # 0 = all
    run(ap.parse_args())
