"""
Paired comparison of two systems that external_test.py scored on the SAME external photos.

    python compare_external.py runs\\idrid_s15 runs\\idrid_s17

Why paired: a small external set gives each system's specificity a wide range on its own
(IDRiD: ~193 healthy photos, about +-6 points). But both systems saw the SAME photos, so
most of that luck is shared. Resampling the photos (by patient) TOGETHER for both systems
and looking at the DIFFERENCE removes the shared part, exactly like analyze.py's paired
comparison on validation.

Each system keeps its own cutoffs (fixed on EyePACS validation by external_test.py, saved
as the referred_standard / referred_cautious columns); nothing is re-chosen here.

Self-check: each system's own numbers are recomputed from its predictions file with the
same resamples (same --n-boot and --seed) and must match its external_results.json exactly.

This only reads finished outputs: it never scores a photo, so it adds no look at a set.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze import REFERABLE_FROM, auc, bootstrap_weights, range95, sens_spec


def load(folder):
    """One external_test.py result folder: its predictions table and its results file."""
    # converters={"system_score": float}: the score comes back exactly as it was saved.
    df = pd.read_csv(os.path.join(folder, "external_predictions.csv"),
                     dtype={"image": str, "patient": str}, converters={"system_score": float})
    with open(os.path.join(folder, "external_results.json")) as f:
        return df, json.load(f)


def points95(diffs):
    """95% range of a difference between two shares, in percentage points."""
    lo, hi = np.percentile(diffs, [2.5, 97.5]) * 100
    return f"[{lo:+.1f} to {hi:+.1f}]"


def self_check(name, sick, df, res, boots):
    """Recompute this system's own numbers; they must equal what external_test.py wrote."""
    score = df["system_score"].to_numpy()
    got = {"auc": auc(sick, score),
           "auc_95": list(np.percentile([auc(sick, score, w) for w in boots], [2.5, 97.5]))}
    for cut in ("standard", "cautious"):
        ref = df[f"referred_{cut}"].to_numpy()
        bs = np.array([sens_spec(sick, ref, w) for w in boots])
        got[cut] = dict(zip(("sensitivity", "specificity"), sens_spec(sick, ref)),
                        specificity_95=list(np.percentile(bs[:, 1], [2.5, 97.5])))
    want = [res["auc"], *res["auc_95"]] + [v for cut in ("standard", "cautious") for v in
                                           (res[cut]["sensitivity"], res[cut]["specificity"],
                                            *res[cut]["specificity_95"])]
    have = [got["auc"], *got["auc_95"]] + [v for cut in ("standard", "cautious") for v in
                                           (got[cut]["sensitivity"], got[cut]["specificity"],
                                            *got[cut]["specificity_95"])]
    if not np.allclose(want, have, rtol=0, atol=1e-9):
        raise SystemExit(f"Self-check FAILED for {name}: recomputed {have} vs external_results.json {want}. "
                         "Different --n-boot/--seed than external_test.py used, or edited files?")
    print(f"  {name}: AUC {got['auc']:.3f} | standard: sens {got['standard']['sensitivity']:.1%}, "
          f"spec {got['standard']['specificity']:.1%} | cautious: sens {got['cautious']['sensitivity']:.1%}, "
          f"spec {got['cautious']['specificity']:.1%}   (matches external_results.json)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="external_test.py output folder of system A (e.g. runs/idrid_s15)")
    p.add_argument("b", help="external_test.py output folder of system B (e.g. runs/idrid_s17)")
    p.add_argument("--n-boot", type=int, default=2000, help="must equal external_test.py's (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="must equal external_test.py's (default 0)")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    args = p.parse_args()

    (da, ra), (db, rb) = load(args.a), load(args.b)
    same = (len(da) == len(db) and (da["image"].values == db["image"].values).all()
            and (da["patient"].values == db["patient"].values).all()
            and (da["level"].values == db["level"].values).all())
    if not same:
        raise SystemExit("The two folders did not score the same photos (same images, patients, grades, "
                         "same order). A paired comparison needs exactly the same external set.")
    sick = (da["level"] >= REFERABLE_FROM).to_numpy()
    # The same resamples external_test.py used (same patients in the same order, same seed).
    boots = list(bootstrap_weights(da["patient"], args.n_boot, args.seed))

    title = (f"Paired external comparison: {len(da)} photos, {da['patient'].nunique()} patients, "
             f"{sick.sum()} referable, {(~sick).sum()} not referable")
    print(title + "\n" + "-" * len(title))
    self_check(f"A ({args.a})", sick, da, ra, boots)
    self_check(f"B ({args.b})", sick, db, rb, boots)

    sa, sb = da["system_score"].to_numpy(), db["system_score"].to_numpy()
    d_auc = np.array([auc(sick, sb, w) - auc(sick, sa, w) for w in boots])
    print(f"\nB - A, each system at its own cutoff (95% range by patient, {args.n_boot} resamples):")
    print(f"  AUC          {auc(sick, sb) - auc(sick, sa):+.4f}   {range95(d_auc)}   "
          f"B better in {np.mean(d_auc > 0):.0%}")
    verdict = {}
    for cut in ("standard", "cautious"):
        ref_a, ref_b = da[f"referred_{cut}"].to_numpy(), db[f"referred_{cut}"].to_numpy()
        diff = np.array([np.subtract(sens_spec(sick, ref_b, w), sens_spec(sick, ref_a, w)) for w in boots])
        d_sens, d_spec = np.subtract(sens_spec(sick, ref_b), sens_spec(sick, ref_a))
        print(f"  {cut:8s} sensitivity {d_sens * 100:+.1f} points {points95(diff[:, 0])} | "
              f"specificity {d_spec * 100:+.1f} points {points95(diff[:, 1])} | "
              f"referred {(ref_b.mean() - ref_a.mean()) * 100:+.1f} points")
        verdict[cut] = (np.percentile(diff[:, 1], 2.5), sens_spec(sick, ref_b)[0])

    lo, sens_b = verdict["standard"]
    gain = lo > 0
    kept = sens_b >= args.target_sensitivity
    print("\nThe exp17/IDRiD rule (standard cutoff):")
    print(f"  1. Specificity gain beyond resampling luck: {'YES' if gain else 'NO'} "
          f"(lower end of the 95% range {lo * 100:+.1f} points)")
    print(f"  2. B still catches >= {args.target_sensitivity:.0%} of sick photos: "
          f"{'YES' if kept else 'NO'} ({sens_b:.1%})")
    print(f"  => {'B shows a smaller camera shift.' if gain and kept else 'No smaller shift shown for B.'}")


if __name__ == "__main__":
    main()
