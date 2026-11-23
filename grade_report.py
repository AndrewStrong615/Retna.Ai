"""
How well does the model tell the five DR grades apart, not just "refer / don't refer"?

The ordinal model (--target ordinal, the default since exp3) learns one severity score s per
eye and four cutpoints, one per grade boundary: P(grade > k) = sigmoid(s + cut[k]). So the
same score that decides referral also gives a full 0-4 grade, at no extra cost:
predicted grade = how many boundaries the eye is more likely than not to be past
(CumulativeLogitLoss.predicted_grade in train.py). This script reads a finished run's
predictions and reports that grade against the true one:

  - the confusion table (rows = true grade, columns = predicted grade);
  - exact agreement and agreement within one grade;
  - quadratic weighted kappa, the official metric of the 2015 Kaggle EyePACS competition:
    1 = perfect, 0 = no better than chance, and a guess 3 grades off costs 9x a guess
    1 grade off. Hand-written below (weighted_kappa), with a 95% range by patient bootstrap.

Nothing is chosen here: the cutpoints come from training (saved in best.pt), so the grade is
fixed before validation is read. Validation by default; --split test only to REPORT a frozen
run once, never to choose.

Works on a training run (runs\\exp15) and on its TTA folder (runs\\exp15_tta): TTA averages
raw scores of the same model, so the same cutpoints apply. NOT on an ensemble or a both-eyes
blend: those scores are standardized, a different scale from the cutpoints.

Caveat: the cutpoints follow the TRAINING grade mix (with DDR or other extra data included).
A dataset with a different mix shifts how often each grade is predicted.

Usage:
  python grade_report.py runs\\exp15_tta
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from analyze import GRADE_NAMES, bootstrap_weights, load_predictions
from external_test import source_run
from train import NUM_GRADES, CumulativeLogitLoss


def load_cutpoints(run_dir):
    """The run's learned cutpoints, rebuilt from best.pt (ordinal runs only)."""
    run = run_dir if os.path.exists(os.path.join(run_dir, "best.pt")) else source_run(run_dir)
    ckpt = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=True)
    if ckpt.get("target", "binary") != "ordinal":
        raise SystemExit(f"{run} was trained with --target {ckpt.get('target')}: only the ordinal target "
                         "has grade cutpoints.")
    loss = CumulativeLogitLoss(list(range(NUM_GRADES)))   # any grades: the saved values replace the start
    loss.load_state_dict(ckpt["loss"])
    return loss, run


def weighted_kappa(true, pred, weight=None, k=NUM_GRADES):
    """
    Quadratic weighted kappa (Cohen 1968), written out:
      O[i, j] = (weighted) share of eyes with true grade i and predicted grade j;
      E[i, j] = what O would be by chance: share with true i times share predicted j;
      W[i, j] = (i - j)^2 / (k - 1)^2, the cost of each kind of mistake;
      kappa   = 1 - sum(W * O) / sum(W * E).
    """
    w = np.ones(len(true)) if weight is None else np.asarray(weight, dtype=float)
    obs = np.zeros((k, k))
    np.add.at(obs, (np.asarray(true), np.asarray(pred)), w)
    obs /= obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    cost = (i - j) ** 2 / (k - 1) ** 2
    denom = (cost * expected).sum()
    return float(1 - (cost * obs).sum() / denom) if denom > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="an ordinal run folder or its <run>_tta folder")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = load_predictions(args.run, args.split)
    loss, run = load_cutpoints(args.run)
    with torch.no_grad():
        pred = loss.predicted_grade(torch.tensor(df["score"].to_numpy(dtype=np.float64))).numpy()
        cuts = loss.cutpoints().numpy()
    true = df["level"].to_numpy()

    print(f"{args.run} ({args.split}, {len(df)} eyes); cutpoints from {run}: "
          + ", ".join(f"grade > {k} at score {-c:+.2f}" for k, c in enumerate(cuts)))
    table = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="predicted")) \
        .reindex(index=range(NUM_GRADES), columns=range(NUM_GRADES), fill_value=0)
    print("\nConfusion table (rows: true grade, columns: predicted grade):")
    print("            " + "".join(f"{c:>8d}" for c in table.columns) + "   recall")
    for g, row in table.iterrows():
        recall = row[g] / row.sum() if row.sum() else float("nan")
        print(f"  {g} {GRADE_NAMES[g]:13s}" + "".join(f"{v:8d}" for v in row) + f"   {recall:6.1%}")

    exact, within1 = np.mean(pred == true), np.mean(np.abs(pred - true) <= 1)
    kappa = weighted_kappa(true, pred)
    boots = [weighted_kappa(true, pred, w) for w in bootstrap_weights(df["patient"], args.n_boot, args.seed)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nExact grade {exact:.1%} | within one grade {within1:.1%} | "
          f"quadratic weighted kappa {kappa:.3f} [95% range {lo:.3f} to {hi:.3f}]")
    # Referral read off the grade (grade >= 2) vs the run's own 90%-sensitivity cutoff: the grade
    # uses the training cutpoint, which sits at P(referable) = 0.5, so it refers fewer eyes.
    sick = true >= 2
    by_grade = pred >= 2
    print(f"Referral from the predicted grade (>= 2): sensitivity {by_grade[sick].mean():.1%}, "
          f"specificity {(~by_grade[~sick]).mean():.1%} (the run's own cutoff: "
          f"{df['referred'][sick].mean():.1%} / {(~df['referred'][~sick]).mean():.1%})")
    if args.split == "test":
        print("\nTest split: a report of a frozen run, never a reason to change anything.")


if __name__ == "__main__":
    main()
