"""
Does a patient's OTHER eye carry extra information, beyond what the scores already say?

The idea ("the fellow eye as each patient's own control"): diabetic retinopathy
usually affects both eyes about equally, and both photos of one person share
everything that is NOT disease (skin pigment, age, lens, camera, exposure). So a
MISMATCH between the two eyes might be a disease signal of its own.

This is a quick kill test with finished models. Nothing is trained, and only
VALIDATION eyes are read (the test set stays untouched; training eyes would be
dishonest, because the model has memorized part of them).

What the project already uses: analyze.py --both-eyes blends each eye's score
with its fellow's (0.6 * own + 0.4 * fellow). That is a weighted AVERAGE. The
question here is different: does the DIFFERENCE between the eyes add anything on
top of the two scores? Three mismatch measures:
  gap        |own score - fellow score|                (needs only predictions files)
  feat dist  distance between the two eyes' 512 pooled features of the DR model
  pre dist   the same with the self-supervised encoder (pretrain.py), which never saw a grade

Two steps:

  1. features (optional, needs the photos and the weights; ~20 min on 4 CPU cores):
       python fellow_eye_check.py features --run runs/exp15 --images <its photos> \\
              [--encoder runs/pre1/encoder.pt] --out runs/fellow_exp15/features.npz
     Each eye is seen as is and mirrored left-right, and the two feature vectors are
     averaged, so that a left and a right eye (mirror images of each other) compare fairly.
     Self-check: the plain view must reproduce the run's own validation AUC.

  2. analyze (seconds, no GPU):
       python fellow_eye_check.py analyze --pred runs/exp15_tta \\
              [--features runs/fellow_exp15/features.npz] --out runs/fellow_exp15

The four tests (all ranges: 95%, bootstrap by PATIENT; fixed before any real number was seen):
  T1  Sicker eye. In patients whose eyes have different grades, how often does the
      higher-scoring eye have the higher grade? Compared with ranking two eyes of the
      same two grades taken from DIFFERENT patients. If "within one patient" is
      clearly better, the fellow eye works as a control (the person's own offset cancels).
  T2  Per eye, referable or not. Small logistic models, fitted on 4/5 of the patients
      and scored on the remaining 1/5 (5 folds, split by patient):
        A own score | B own + fellow (the blend) | C = B + gap | D = B + gap + feature distances
        E a flexible model (boosted trees) on own + fellow: ANY function of the two scores.
      A mismatch measure "adds signal" only if AUC(C or D) - AUC(B) has a range above 0.
  T3  Per patient, "any eye referable". P0 = the worse eye's score alone; P1 = worse eye + gap;
      P2 = P1 + feature distances.
  T4  Different grades or not. How well does each mismatch measure, alone, find the
      patients whose two eyes were graded differently? (The label-free flavor of the idea.)

Several tests on the same 2,600 patients: one lucky "range above 0" out of many is
not a finding. A real effect should show in T2 AND T3, and with more than one model.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REFERABLE_FROM = 2


# ----------------------------------------------------------------------------
# Step 1: features
# ----------------------------------------------------------------------------

def extract_features(args):
    # torch is imported here so that `analyze` runs on a machine without it.
    import torch
    from torch.utils.data import DataLoader

    from data import FundusDataset, make_transforms
    from model import build_model

    source = os.path.join(args.run, "val_predictions.csv")    # validation only, on purpose
    eyes = pd.read_csv(source, dtype={"image": str, "patient": str})[["image", "patient", "level"]]
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")
    arch = ckpt.get("arch", {})
    kind = dict(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"), dropout=arch.get("dropout", 0.2))
    model = build_model(**kind)
    model.load_state_dict(ckpt["model"])
    nets = {"model": model.eval()}
    if args.encoder:
        # The self-supervised encoder: same network, stem and stages only (as train.py --init).
        # Its pooling layer was never trained, so it keeps the starting value (GeM p = 3).
        pre = torch.load(args.encoder, map_location="cpu")
        enc = build_model(depth=pre["arch"]["depth"], pool=pre["arch"]["pool"])
        missing, unexpected = enc.load_state_dict(pre["encoder"], strict=False)
        if unexpected or [k for k in missing if not k.startswith(("head.", "pool."))]:
            raise SystemExit(f"{args.encoder} does not fit the network.")
        nets["pre"] = enc.eval()

    _, tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])
    loader = DataLoader(FundusDataset(eyes, args.images, tf), batch_size=args.batch, num_workers=args.workers)
    torch.set_num_threads(os.cpu_count() or 1)
    feats = {k: [] for k in nets}
    plain = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            for name, net in nets.items():
                # As is and mirrored: a right eye is roughly a mirrored left eye.
                f = [net.pool(net.stages(net.stem(v))).flatten(1) for v in (x, x.flip(3).contiguous())]
                feats[name].append(((f[0] + f[1]) / 2).numpy())
                if name == "model":
                    plain.append(net.head(f[0]).squeeze(1).numpy())   # the run's own plain-view score
            if i % 10 == 0:
                print(f"  {min((i + 1) * args.batch, len(eyes))}/{len(eyes)} eyes", flush=True)
    plain = np.concatenate(plain)
    check = roc_auc_score(eyes["level"] >= REFERABLE_FROM, plain)
    print(f"check: plain view AUC {check:.4f} (run's own best val AUC {ckpt.get('val_auc', float('nan')):.4f})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, image=eyes["image"].to_numpy().astype(str), plain=plain,
                        **{f"feat_{k}": np.concatenate(v) for k, v in feats.items()})
    print(f"Saved {args.out}")


# ----------------------------------------------------------------------------
# Step 2: analysis
# ----------------------------------------------------------------------------

def pair_table(pred_dir, features):
    """
    One row per eye, only patients with exactly two photos, each row also carrying
    the fellow eye's score and grade and the mismatch measures of its patient.
    """
    df = pd.read_csv(os.path.join(pred_dir, "val_predictions.csv"),
                     dtype={"image": str, "patient": str}, converters={"score": float})
    df = df[["image", "patient", "level", "score"]]
    n_all = df["patient"].nunique()
    df = df[df.groupby("patient")["image"].transform("count") == 2].sort_values(["patient", "image"])
    df = df.reset_index(drop=True)
    print(f"{pred_dir}: {df['patient'].nunique()} patients with both eyes (of {n_all}), {len(df)} eyes")
    # Rows come in pairs (2k, 2k+1): swapping neighbours gives each eye its fellow.
    swap = np.arange(len(df)).reshape(-1, 2)[:, ::-1].ravel()
    df["fellow"] = df["score"].to_numpy()[swap]
    df["fellow_level"] = df["level"].to_numpy()[swap]
    df["gap"] = (df["score"] - df["fellow"]).abs()
    df["sick"] = df["level"] >= REFERABLE_FROM
    measures = ["gap"]
    if features:
        z = np.load(features)
        order = pd.Series(np.arange(len(z["image"])), index=z["image"]).reindex(df["image"])
        if order.isna().any():
            raise SystemExit(f"{features} lacks {int(order.isna().sum())} of the eyes in {pred_dir}.")
        order = order.to_numpy().astype(int)
        for key in [k for k in z.files if k.startswith("feat_")]:
            f = z[key][order].astype(np.float64)
            # Each of the 512 features on the same scale (mean 0, spread 1 over these eyes;
            # no grade is used), then the angle between the two eyes' vectors:
            # cosine distance, 0 = same direction, 1 = unrelated.
            f = (f - f.mean(0)) / (f.std(0) + 1e-9)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
            name = "dist_" + key[len("feat_"):]
            df[name] = 1.0 - (f * f[swap]).sum(1)
            measures.append(name)
    return df, measures


def patient_bootstrap(df, n_boot, seed):
    """Per round: how many times each row counts (its patient's number of draws)."""
    codes, uniques = pd.factorize(df["patient"])
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength=len(uniques))[codes]


def summary(point, boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": float(point), "low": float(lo), "high": float(hi)}


def show(label, s, percent=False):
    f = (lambda v: f"{v:.1%}") if percent else (lambda v: f"{v:+.4f}" if abs(s["value"]) < 0.3 else f"{v:.4f}")
    print(f"  {label:<58} {f(s['value'])}  [{f(s['low'])} to {f(s['high'])}]")


def wauc(y, x, w):
    keep = w > 0
    return roc_auc_score(y[keep], x[keep], sample_weight=w[keep])


def test_sicker_eye(df, n_boot, seed):
    """T1: within a patient vs between patients, for the same pairs of grades."""
    first, second = df.iloc[0::2].reset_index(drop=True), df.iloc[1::2].reset_index(drop=True)
    differ = (first["level"] != second["level"]).to_numpy()
    lo_grade = np.minimum(first["level"], second["level"]).to_numpy()
    hi_grade = np.maximum(first["level"], second["level"]).to_numpy()
    # +1 if the higher-graded eye has the higher score, 0.5 for a tie in score.
    d = np.where(first["level"] > second["level"], 1, -1) * (first["score"] - second["score"]).to_numpy()
    correct = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
    level, score = df["level"].to_numpy(), df["score"].to_numpy()

    def both(pw, ew):
        """pw: weight per patient; ew: weight per eye. Returns (within, between)."""
        w = pw * differ
        within = (w * correct).sum() / w.sum()
        # Between patients: for each grade pair (a < b), the chance that a random
        # grade-b eye outscores a random grade-a eye (that is an AUC), averaged with
        # the same grade-pair mix as the discordant patients.
        total = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                share = w[(lo_grade == a) & (hi_grade == b)].sum()
                if share == 0:
                    continue
                pick = ((level == a) | (level == b)) & (ew > 0)
                total += share * roc_auc_score(level[pick] == b, score[pick], sample_weight=ew[pick])
        return within, total / w.sum()

    ones_p, ones_e = np.ones(len(first)), np.ones(len(df))
    w0, b0 = both(ones_p, ones_e)
    ws, bs = [], []
    for ew in patient_bootstrap(df, n_boot, seed):
        w, b = both(ew[0::2].astype(float), ew.astype(float))
        ws.append(w), bs.append(b)
    ws, bs = np.array(ws), np.array(bs)
    out = {"discordant_patients": int(differ.sum()), "within_patient": summary(w0, ws),
           "between_patients": summary(b0, bs), "within_minus_between": summary(w0 - b0, ws - bs)}
    print(f"\nT1 sicker eye: {differ.sum()} of {len(first)} patients have eyes with different grades")
    show("higher score = higher grade, within one patient", out["within_patient"], percent=True)
    show("same grade pairs, eyes from different patients", out["between_patients"], percent=True)
    show("within - between", out["within_minus_between"])
    return out


def out_of_fold(X, y, groups, flexible=False, folds=5, seed=0):
    """
    Predictions for every row from a model that never saw that row's PATIENT:
    patients are dealt into 5 folds; each fold is scored by a model fitted on the other 4.
    """
    codes, uniques = pd.factorize(groups)
    fold_of = np.random.default_rng(seed).permutation(len(uniques)) % folds
    fold = fold_of[codes]
    pred = np.zeros(len(y))
    for k in range(folds):
        if flexible:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))
        m.fit(X[fold != k], y[fold != k])
        # The log-odds, so that the 5 folds' outputs sit on one comparable scale.
        p = np.clip(m.predict_proba(X[fold == k])[:, 1], 1e-9, 1 - 1e-9)
        pred[fold == k] = np.log(p / (1 - p))
    return pred


def compare_models(df, y, models, base, n_boot, seed, flexible=()):
    """Out-of-fold AUC of each feature set, and its paired difference from `base`."""
    preds = {name: out_of_fold(df[cols].to_numpy(), y, df["patient"], flexible=name in flexible)
             for name, cols in models.items()}
    point = {n: roc_auc_score(y, p) for n, p in preds.items()}
    boots = {n: [] for n in preds}
    for w in patient_bootstrap(df, n_boot, seed):
        for n, p in preds.items():
            boots[n].append(wauc(y, p, w))
    boots = {n: np.array(b) for n, b in boots.items()}
    out = {}
    for n in preds:
        out[n] = {"auc": summary(point[n], boots[n])}
        show(f"{n}: AUC", out[n]["auc"])
        if n != base:
            out[n]["minus_" + base[:2].strip()] = summary(point[n] - point[base], boots[n] - boots[base])
            show(f"    difference from {base[:2].strip()}", out[n]["minus_" + base[:2].strip()])
    return out


def analyze(args):
    df, measures = pair_table(args.pred, args.features)
    dists = [m for m in measures if m != "gap"]
    results = {"pred": os.path.abspath(args.pred), "features": args.features, "eyes": len(df),
               "patients": int(df["patient"].nunique()), "n_boot": args.n_boot}

    results["T1"] = test_sicker_eye(df, args.n_boot, args.seed)

    # ---- T2: per eye ----
    y = df["sick"].to_numpy()
    print(f"\nT2 per eye, referable or not ({y.sum()} of {len(y)} eyes referable), out-of-fold AUC")
    models = {"A own score": ["score"], "B own + fellow": ["score", "fellow"],
              "C B + gap": ["score", "fellow", "gap"],
              "E flexible(own, fellow)": ["score", "fellow"]}
    if dists:
        models["D B + gap + feature distances"] = ["score", "fellow", "gap"] + dists
        models["F flexible(all)"] = ["score", "fellow", "gap"] + dists
    results["T2"] = compare_models(df, y, models, "B own + fellow", args.n_boot, args.seed,
                                   flexible=("E flexible(own, fellow)", "F flexible(all)"))

    # ---- T3: per patient ----
    g = df.groupby("patient", sort=False)
    pat = pd.DataFrame({"patient": g["patient"].first(), "worse": g["score"].max(), "gap": g["gap"].first(),
                        "any_sick": g["sick"].max(), "differ": g["level"].nunique() > 1,
                        "ref_differ": g["sick"].nunique() > 1,
                        **{d: g[d].first() for d in dists}}).reset_index(drop=True)
    yp = pat["any_sick"].to_numpy()
    print(f"\nT3 per patient, any eye referable ({yp.sum()} of {len(yp)} patients), out-of-fold AUC")
    models = {"P0 worse eye's score": ["worse"], "P1 worse eye + gap": ["worse", "gap"]}
    if dists:
        models["P2 P1 + feature distances"] = ["worse", "gap"] + dists
    results["T3"] = compare_models(pat, yp, models, "P0 worse eye's score", args.n_boot, args.seed)

    # ---- T4: each mismatch measure alone ----
    print("\nT4 each mismatch measure alone (AUC; 0.5 = no information)")
    results["T4"] = {}
    targets = {"eyes graded differently": "differ", "one eye referable, the other not": "ref_differ",
               "any eye referable": "any_sick"}
    for label, col in targets.items():
        t = pat[col].to_numpy()
        print(f"  target: {label} ({t.sum()} of {len(t)} patients)")
        for m in measures:
            x = pat[m].to_numpy()
            boots = [wauc(t, x, w) for w in patient_bootstrap(pat, args.n_boot, args.seed)]
            results["T4"][f"{col}|{m}"] = summary(roc_auc_score(t, x), boots)
            show(f"    {m}", results["T4"][f"{col}|{m}"])

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(os.path.normpath(args.pred))
        with open(os.path.join(args.out, f"fellow_eye_{name}.json"), "w") as f:
            json.dump(results, f, indent=2)
        df.to_csv(os.path.join(args.out, f"fellow_eye_{name}_eyes.csv"), index=False)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--run", required=True, help="a finished run, e.g. runs/exp15 (best.pt, val_predictions.csv)")
    f.add_argument("--images", required=True, help="the preprocessed photos that run was trained on")
    f.add_argument("--encoder", default=None, help="optional: pretrain.py's encoder.pt (label-free features)")
    f.add_argument("--out", required=True, help="features file to write (.npz)")
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--workers", type=int, default=2)
    a = sub.add_parser("analyze")
    a.add_argument("--pred", required=True, help="folder with val_predictions.csv (a run or its _tta folder)")
    a.add_argument("--features", default=None, help="features.npz from the features step (optional)")
    a.add_argument("--out", default=None, help="folder for the results JSON and the per-eye table")
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    extract_features(args) if args.cmd == "features" else analyze(args)


if __name__ == "__main__":
    main()
