"""
Analyze finished runs from their saved predictions (no GPU, no images needed).

One run: how good is it, and how sure can we be?
  python analyze.py runs\\exp6
    - AUC, sensitivity and specificity on the validation eyes, each with a 95%
      range (bootstrap, see below)
    - how often each grade 0-4 gets referred (does it catch the severe eyes?)
    - per patient: refer a person if EITHER eye is flagged (how screening works)

Two runs: is B really better than A, or just luckier?
  python analyze.py runs\\exp6 runs\\exp7
    - both models scored the SAME validation eyes, so we compare them eye by eye
      ("paired"). Most of the luck of which eyes are easy cancels out, which makes
      this far more sensitive than comparing two AUC numbers from a table.

The bootstrap, in plain words: we don't have a second validation set, so we
fake many. Draw patients at random, with replacement, until we have as many as
the real set (some patients appear twice, some not at all), and recompute the
number. Do that 2000 times; the middle 95% of the results is the range. We draw
PATIENTS, not eyes, because a person's two eyes are not independent.

What the range does NOT include: training luck. Training the same recipe again
with another --seed also moves the score (run-to-run noise). Once you have such a
repeat run, pass the difference you saw as --noise-floor and the verdict uses it.

Two free tricks (no retraining), decided on validation only:
  python analyze.py runs\\exp8 runs\\exp9 --ensemble      average several models
  python analyze.py runs\\exp8 --both-eyes                  use the fellow eye too
  python analyze.py runs\\exp8 runs\\exp9 --ensemble --both-eyes
  ... add --final-test to apply exactly what validation chose to the test set ONCE,
      for the frozen result. Never use that test number to change the choice.

Rules (CLAUDE.md): decisions use validation only. A comparison of two runs is
always on validation. `--split test` is allowed for ONE run, to report the final
numbers with ranges for the record; never use it to choose between runs.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REFERABLE_FROM = 2
GRADE_NAMES = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "proliferative"}


def load_predictions(run, split):
    """One row per eye: image, patient, level, score, referred."""
    path = os.path.join(run, f"{split}_predictions.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Runs made before the revamp (exp1, exp2) have no "
                         "predictions files.")
    # converters={"score": float}: Python's float() turns the saved text back into exactly the
    # same number (pandas' own float_precision option is deprecated in newer versions).
    df = pd.read_csv(path, dtype={"image": str, "patient": str}, converters={"score": float})
    if "referred" not in df:                       # early files: rebuild the decision
        with open(os.path.join(run, "test_results.json")) as f:
            df["referred"] = df["score"] >= json.load(f)["threshold"]
    df["sick"] = df["level"] >= REFERABLE_FROM
    return df


def bootstrap_weights(patients, n_boot, seed):
    """
    Yields, for each bootstrap round, how many times each eye is counted: an eye
    is counted as often as its patient was drawn (one round at a time, to save memory).
    """
    codes, uniques = pd.factorize(patients)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniques), size=len(uniques))
        yield np.bincount(draw, minlength=len(uniques))[codes]


def auc(sick, score, weight=None):
    return float(roc_auc_score(sick, score, sample_weight=weight))


def sens_spec(sick, referred, weight=None):
    w = np.ones(len(sick)) if weight is None else weight
    sens = (w * (referred & sick)).sum() / max((w * sick).sum(), 1e-9)
    spec = (w * (~referred & ~sick)).sum() / max((w * ~sick).sum(), 1e-9)
    return float(sens), float(spec)


def range95(values, percent=False):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"[{lo:.1%} to {hi:.1%}]" if percent else f"[{lo:.3f} to {hi:.3f}]"


def report_one(run, split, n_boot, seed):
    df = load_predictions(run, split)
    sick, score, referred = df["sick"].to_numpy(), df["score"].to_numpy(), df["referred"].to_numpy()
    title = f"{run} | {split} | {len(df)} eyes, {df['patient'].nunique()} patients"
    print(title + "\n" + "-" * len(title))
    if split == "test":
        print("TEST SET: for the record only. Never use these numbers to choose a run or a setting.\n")

    boot_auc, boot_ss = [], []
    for w in bootstrap_weights(df["patient"], n_boot, seed):
        boot_auc.append(auc(sick, score, w))
        boot_ss.append(sens_spec(sick, referred, w))
    boot_ss = np.array(boot_ss)
    sens, spec = sens_spec(sick, referred)
    print("Per eye (cutoff chosen on validation for 90% sensitivity):")
    print(f"  AUC          {auc(sick, score):.3f}   95% range {range95(boot_auc)}")
    print(f"  sensitivity  {sens:.1%}   95% range {range95(boot_ss[:, 0], True)}  (sick eyes referred)")
    print(f"  specificity  {spec:.1%}   95% range {range95(boot_ss[:, 1], True)}  (healthy eyes cleared)")
    if split == "val":
        print("  (validation sensitivity/specificity are a little optimistic: the cutoff was picked on these eyes)")

    print("\nShare referred, by true grade (we want low for 0-1, high for 2-4):")
    for g, part in df.groupby("level"):
        print(f"  grade {g} ({GRADE_NAMES.get(g, '?'):13s}) {part['referred'].mean():6.1%} of {len(part):5d} eyes")

    # Screening refers PEOPLE: a patient is referred if either eye is flagged,
    # and is truly referable if either eye is grade 2 or worse.
    per_patient = df.groupby("patient").agg(sick=("sick", "max"), referred=("referred", "max"),
                                            score=("score", "max"))
    p_sens, p_spec = sens_spec(per_patient["sick"].to_numpy(), per_patient["referred"].to_numpy())
    print(f"\nPer patient (refer if either eye is flagged), {len(per_patient)} patients:")
    print(f"  AUC {auc(per_patient['sick'], per_patient['score']):.3f} (using the higher of the two eye scores)"
          f" | sensitivity {p_sens:.1%} | specificity {p_spec:.1%}")


def compare(run_a, run_b, n_boot, seed, noise_floor):
    a, b = load_predictions(run_a, "val"), load_predictions(run_b, "val")
    merged = a.merge(b, on="image", suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise SystemExit(f"The runs did not score the same validation eyes ({len(a)} vs {len(b)}, "
                         f"{len(merged)} shared): different --split-seed or missing images. "
                         "A paired comparison needs the same eyes.")
    if not (merged["level_a"] == merged["level_b"]).all():
        raise SystemExit("The same eyes have different labels in the two runs: different label files?")

    sick = merged["sick_a"].to_numpy()
    sa, sb = merged["score_a"].to_numpy(), merged["score_b"].to_numpy()
    ra, rb = merged["referred_a"].to_numpy(), merged["referred_b"].to_numpy()
    diffs = np.array([auc(sick, sb, w) - auc(sick, sa, w)
                      for w in bootstrap_weights(merged["patient_a"], n_boot, seed)])
    d = auc(sick, sb) - auc(sick, sa)

    title = f"Paired comparison on the same {len(merged)} validation eyes"
    print(title + "\n" + "-" * len(title))
    print(f"  AUC  A ({run_a}): {auc(sick, sa):.4f}")
    print(f"  AUC  B ({run_b}): {auc(sick, sb):.4f}")
    print(f"  B - A: {d:+.4f}   95% range {range95(diffs)}   B better in {np.mean(diffs > 0):.0%} of resamples")
    for name, r in (("A", ra), ("B", rb)):
        s, p = sens_spec(sick, r)
        print(f"  {name} at its own cutoff: sensitivity {s:.1%}, specificity {p:.1%}")

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    beyond_sampling = lo > 0 or hi < 0
    print("\nVerdict (a difference counts as real only if it passes BOTH checks):")
    print(f"  1. Luck of which patients are in validation: "
          f"{'passed, the 95% range excludes 0' if beyond_sampling else 'NOT passed, the 95% range includes 0'}")
    if noise_floor is not None:
        beyond_noise = abs(d) > noise_floor
        print(f"  2. Training luck: {'passed' if beyond_noise else 'NOT passed'}, the difference {abs(d):.4f} is "
              f"{'larger' if beyond_noise else 'not larger'} than the run-to-run noise {noise_floor:.4f}")
        if beyond_sampling and beyond_noise:
            print(f"  => {'B' if d > 0 else 'A'} is really better.")
        else:
            print("  => Call it a tie for now.")
    else:
        print("  2. Training luck: unknown. Another --seed also moves the score; measure it once with a\n"
              "     seed-repeat run and pass that difference as --noise-floor.")
        print(f"  => {('Probably ' + ('B' if d > 0 else 'A') + ' is better, but check 2 is still open.') if beyond_sampling else 'Call it a tie for now.'}")


def pick_cutoff(sick, score, target_sensitivity):
    """The highest cutoff that still refers at least target_sensitivity of the sick eyes
    (same rule as train.py)."""
    _, tpr, thresholds = roc_curve(sick, score, drop_intermediate=False)
    return float(thresholds[np.where(tpr >= target_sensitivity)[0][0]])


def load_aligned(runs, split):
    """Predictions of several runs on the same eyes, as one table (columns score_0, score_1, ...)."""
    tables = [load_predictions(r, split) for r in runs]
    base = tables[0][["image", "patient", "level", "sick"]].copy()
    for i, t in enumerate(tables):
        merged = base.merge(t[["image", "level", "score"]], on="image", how="inner")
        if len(merged) != len(base) or len(t) != len(base) or not (merged["level_x"] == merged["level_y"]).all():
            raise SystemExit(f"{runs[i]} did not score the same {split} eyes with the same labels as {runs[0]}.")
        base[f"score_{i}"] = merged["score"].to_numpy()
    return base


def fellow_eye(df, score):
    """For each eye, the score of the patient's OTHER eye (its own score if the patient has one photo)."""
    s = pd.Series(score, index=df.index)
    total = s.groupby(df["patient"]).transform("sum")
    count = s.groupby(df["patient"]).transform("count")
    return np.where(count == 2, total - s, s).astype(float)


def tricks(runs, ensemble, both_eyes, final_test, target_sensitivity, n_boot, seed):
    """
    Everything is CHOSEN on validation: each model's score scale (for averaging),
    the fellow-eye weight w, and the 90%-sensitivity cutoff. With --final-test the
    same choices are then applied, once, to the test eyes.
    """
    val = load_aligned(runs, "val")
    n = len(runs) if ensemble else 1

    # 1. Ensemble: put every model's scores on the same scale (standardize with
    #    that model's VALIDATION mean and spread), then average them.
    stats = [(val[f"score_{i}"].mean(), val[f"score_{i}"].std()) for i in range(n)]

    def combine(table):
        return np.mean([(table[f"score_{i}"] - m) / sd for i, (m, sd) in enumerate(stats)], axis=0)

    sick, single = val["sick"].to_numpy(), ((val["score_0"] - stats[0][0]) / stats[0][1]).to_numpy()
    combined = combine(val)

    # 2. Both eyes: score = (1 - w) * own + w * other eye; w chosen on validation.
    w_best = 0.0
    if both_eyes:
        other = fellow_eye(val, combined)
        grid = np.round(np.arange(0, 0.55, 0.05), 2)
        aucs = [auc(sick, (1 - w) * combined + w * other) for w in grid]
        w_best = float(grid[int(np.argmax(aucs))])
        print("Fellow-eye weight w, validation AUC for each (w = 0 means each eye alone):")
        print("  " + "  ".join(f"{w:.2f}:{a:.4f}" for w, a in zip(grid, aucs)))
        print(f"  chosen w = {w_best:.2f}  (picked on validation, so its validation AUC is a little optimistic)\n")
        combined = (1 - w_best) * combined + w_best * other

    name = " + ".join(["ensemble of " + ", ".join(runs) if n > 1 else runs[0]] +
                      (["both eyes"] if both_eyes else []))
    title = f"Validation: {name} vs {runs[0]} alone ({len(val)} eyes)"
    print(title + "\n" + "-" * len(title))
    diffs = np.array([auc(sick, combined, wt) - auc(sick, single, wt)
                      for wt in bootstrap_weights(val["patient"], n_boot, seed)])
    cut = pick_cutoff(sick, combined, target_sensitivity)
    sens, spec = sens_spec(sick, combined >= cut)
    s1, p1 = sens_spec(sick, single >= pick_cutoff(sick, single, target_sensitivity))
    print(f"  AUC {runs[0]} alone: {auc(sick, single):.4f} | sensitivity {s1:.1%}, specificity {p1:.1%}")
    print(f"  AUC with the trick(s): {auc(sick, combined):.4f} | sensitivity {sens:.1%}, specificity {spec:.1%}")
    print(f"  difference {auc(sick, combined) - auc(sick, single):+.4f}   95% range {range95(diffs)}"
          f"   better in {np.mean(diffs > 0):.0%} of resamples")

    if not final_test:
        print("\nIf this helps and you want it in the frozen result, rerun with --final-test (once).")
        return
    test = load_aligned(runs, "test")
    t_sick, t_combined = test["sick"].to_numpy(), combine(test)
    if both_eyes:
        t_combined = (1 - w_best) * t_combined + w_best * fellow_eye(test, t_combined)
    t_ref = t_combined >= cut                     # the cutoff chosen on validation
    boot_auc, boot_ss = [], []
    for wt in bootstrap_weights(test["patient"], n_boot, seed):
        boot_auc.append(auc(t_sick, t_combined, wt))
        boot_ss.append(sens_spec(t_sick, t_ref, wt))
    boot_ss = np.array(boot_ss)
    t_sens, t_spec = sens_spec(t_sick, t_ref)
    print(f"\nTEST SET, evaluated once with the choices made on validation ({len(test)} eyes):")
    print("  For the record only: never change a choice because of these numbers.")
    print(f"  AUC          {auc(t_sick, t_combined):.3f}   95% range {range95(boot_auc)}")
    print(f"  sensitivity  {t_sens:.1%}   95% range {range95(boot_ss[:, 0], True)}")
    print(f"  specificity  {t_spec:.1%}   95% range {range95(boot_ss[:, 1], True)}")


def main():
    p = argparse.ArgumentParser(description="Analyze runs from their saved predictions.")
    p.add_argument("runs", nargs="+", help="one run folder (report) or two (paired comparison: A then B)")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="one run only: which predictions to report (test = for the record only)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap rounds")
    p.add_argument("--noise-floor", type=float, default=None,
                   help="AUC difference between two runs that differ only in --seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ensemble", action="store_true", help="average the given runs' scores")
    p.add_argument("--both-eyes", action="store_true", help="blend each eye's score with the fellow eye's")
    p.add_argument("--final-test", action="store_true",
                   help="with --ensemble/--both-eyes: apply the validation choices to the test set once")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    args = p.parse_args()

    if args.ensemble or args.both_eyes:
        if args.ensemble and len(args.runs) < 2:
            raise SystemExit("--ensemble needs at least two runs.")
        if args.split != "val":
            raise SystemExit("Use --final-test (not --split test) to get the test numbers for a trick.")
        tricks(args.runs, args.ensemble, args.both_eyes, args.final_test, args.target_sensitivity,
               args.n_boot, args.seed)
        return
    if args.final_test:
        raise SystemExit("--final-test goes with --ensemble and/or --both-eyes.")
    if len(args.runs) == 1:
        report_one(args.runs[0], args.split, args.n_boot, args.seed)
    elif len(args.runs) == 2:
        if args.split != "val":
            raise SystemExit("Comparisons between runs use validation only (project rule).")
        compare(args.runs[0], args.runs[1], args.n_boot, args.seed, args.noise_floor)
    else:
        raise SystemExit("Give one run (report) or two runs (comparison).")


if __name__ == "__main__":
    main()
