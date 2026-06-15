"""
Analyse Stage 3 results (§4.1/§4.3): mean per arm + 95% bootstrap CI (10k),
Friedman (paired 3 arms) + pairwise Wilcoxon signed-rank + Bonferroni, then a
per-class breakdown for hypotheses H2-H5.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ARMS = ["bbox", "mask", "hybrid"]


def boot_ci(vals, n=10000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_matrix(rows, metric, subset=None):
    by = defaultdict(dict)
    for r in rows:
        if subset and r["pred_cls_name"] != subset:
            continue
        by[r["det_id"]][r["arm"]] = r[metric]
    dets = [d for d in by if all(a in by[d] for a in ARMS)]
    return np.array([[by[d][a] for a in ARMS] for d in dets], float)


def run(args):
    rows = [json.loads(l) for l in open(f"{args.drive}/outputs/metrics/results.jsonl")]
    ok = sum(r["parsed_ok"] for r in rows)
    print(f"rows {len(rows)} | parsed_ok {ok}/{len(rows)}")

    for metric in ["HR", "GS", "CTC"]:
        print(f"\n=== {metric} (HR/CTC lower better, GS higher better) ===")
        for a in ARMS:
            vals = [r[metric] for r in rows if r["arm"] == a]
            m, lo, hi = boot_ci(vals)
            print(f"  {a:7s} mean {m:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  n={len(vals)}")
        M = paired_matrix(rows, metric)
        if M.shape[0] >= 3:
            stat, p = friedmanchisquare(*[M[:, i] for i in range(3)])
            print(f"  Friedman chi2={stat:.3f} p={p:.4f} (n_paired={M.shape[0]})")
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                try:
                    w, pw = wilcoxon(M[:, i], M[:, j])
                    print(f"    {ARMS[i]} vs {ARMS[j]}: W={w:.1f} p={pw:.4f} p_bonf={min(1, pw*3):.4f}")
                except ValueError:
                    print(f"    {ARMS[i]} vs {ARMS[j]}: (no variance / tie)")

    print("\n=== per-class mean (H2-H5) ===")
    classes = sorted(set(r["pred_cls_name"] for r in rows))
    for metric in ["HR", "GS", "CTC"]:
        print(f"\n {metric}:")
        for cls in classes:
            cells = []
            for a in ARMS:
                vals = [r[metric] for r in rows if r["arm"] == a and r["pred_cls_name"] == cls]
                cells.append(f"{a}={np.mean(vals):.3f}" if vals else f"{a}=NA")
            print(f"  {cls:20s} " + "  ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive/opg-live")
    run(ap.parse_args())
