"""
Stage 3 orchestrator: untuk tiap deteksi × 3 arm -> RAG retrieve -> GPT-4o (OpenRouter)
-> parse L-F-V -> metrik HR/GS/CTC. Simpan results.jsonl ke Drive.

RAG chunks DI-RETRIEVE SEKALI per deteksi dan dipakai ulang di 3 arm (held constant,
§4.3). Hanya spatial referent (arm) yang berubah.

Pakai --limit untuk tes murah dulu (mis. --limit 3 = 9 panggilan GPT-4o) sebelum full.
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
    print(f"Deteksi: {len(manifest)} × {len(ARMS)} arm = {len(manifest)*len(ARMS)} panggilan GPT-4o")

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
        # RAG sekali per deteksi (sama untuk 3 arm)
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
            met = explanation_metrics(findings, kb, mask_id, target_fdi=fdi, hr_thr=args.hr_thr)
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
            print(f"  {mi+1}/{len(manifest)} deteksi selesai")

    fout.close()
    print(f"\n✅ {n_done} hasil ({n_fail} gagal) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    ap.add_argument("--kb", default="/content/opg-live/data/kb")
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=4)          # top-k chunk RAG
    ap.add_argument("--hr_thr", type=float, default=0.35)  # lexical support threshold
    ap.add_argument("--limit", type=int, default=0)       # 0 = semua
    run(ap.parse_args())
