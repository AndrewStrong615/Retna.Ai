"""
Pretend clinics: does AdaBN still put a new clinic on the right operating point when the
clinic's DISEASE RATE differs from EyePACS's? (Experiment E0 of PREREG.md.)

Why: on IDRiD, re-measuring the BatchNorm statistics on the clinic's own ungraded photos
(AdaBN) raised specificity from 39.9% to 75.1% at 96.3% sensitivity. But 63% of IDRiD's
photos were sick, and a real screening clinic sees 5-20% sick. BatchNorm statistics
measured on mostly sick photos are not the same as statistics measured on mostly healthy
ones, so AdaBN can move the scores because of the disease rate alone, with no camera
change at all. E0 measures exactly that, using only EyePACS VALIDATION eyes (no camera
change, no external set touched):

  1. The validation patients are split in two halves (both eyes of a patient together).
     Half 0 is the "set-up pool", half 1 the "scored clinic".
  2. A pretend clinic = a set-up batch of --n-setup photos drawn from the pool with a chosen
     share p of sick eyes (grade >= 2). The grades are used ONLY to build the batch, the way
     a real clinic's (unknown) disease rate would be: no method ever reads a grade.
  3. Each set-up method gets the set-up batch, then the scored clinic is scored and judged
     at the cutoff fixed on EyePACS validation (90% sensitivity), never re-chosen.
  4. Repeated --draws times per disease rate, so the spread between draws is measured too:
     a real clinic gets ONE draw, so the bad draws matter as much as the average.

Methods (none reads a grade):
  M0  nothing: the model and cutoff as shipped.
  M1  AdaBN: all BatchNorm layers re-measured on the set-up batch (external_test.adapt_batchnorm).
  M3  score reset: the network is untouched; scores are shifted and scaled so the set-up
      batch's mean and spread match those of all EyePACS validation eyes.
Plus one "fake camera" check at the training disease rate (--fake-rate): every photo
(set-up and scored) is passed through a FIXED camera change (gamma 1.3, Gaussian blur
sigma 1.0, JPEG quality 40; registered in PREREG.md before any run), to see whether AdaBN
undoes a known camera shift at home.

Cost: runs on a GPU when there is one (Kaggle), else on the CPU (cloud, ~20 photos/s), always fp32. Plain view only (no 8-view TTA): each AdaBN
draw needs a full new pass over the scored clinic, and TTA would cost 8x. The cutoff is
therefore the PLAIN-view 90%-sensitivity cutoff, recomputed here on all validation eyes
with this CPU's own scores (self-check: their AUC must match the run's val_predictions.csv).
To save time the scored clinic keeps all of half 1's sick eyes and a fixed random sample of
--n-healthy of its healthy eyes (specificity is estimated without bias, a bit more noisily).
Photos are decoded once and kept in memory as uint8 (about 0.8 MB each at 512 px).

Usage (laptop or cloud; resumes if stopped, writes each finished draw to draws.csv):
    python site_sim.py --run runs/exp15 --images <...>/processed_512_enhanced_tight --out runs/e0_exp15
Outputs in --out: settings.json, draws.csv (one row per method x draw), e0_results.json
(summary per cell + the registered verdicts).
"""

import argparse
import copy
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from scipy.stats import spearmanr

from analyze import REFERABLE_FROM, auc, load_predictions, pick_cutoff, sens_spec
from external_test import adapt_batchnorm, two_halves
from model import build_model

# Registered in PREREG.md before any run: the fixed "fake camera".
FAKE_GAMMA, FAKE_BLUR_SIGMA, FAKE_JPEG_QUALITY = 1.3, 1.0, 40


def fake_camera(img):
    """A fixed, deterministic camera change, in a camera's order: exposure curve (gamma),
    softer optics (blur), then saving the file (JPEG). Same steps as --aug strong, fixed values."""
    lut = [round(255 * (i / 255) ** FAKE_GAMMA) for i in range(256)]
    img = img.point(lut * 3)
    img = img.filter(ImageFilter.GaussianBlur(FAKE_BLUR_SIGMA))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=FAKE_JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class PhotoCache:
    """Every needed photo decoded once, as uint8 (3, size, size) tensors; normalized on the fly
    exactly as data.make_transforms' eval transform does (Resize, ToTensor, Normalize)."""

    def __init__(self, names, images, size, mean, std, camera=None):
        self.index = {n: i for i, n in enumerate(names)}
        self.data = torch.empty((len(names), 3, size, size), dtype=torch.uint8)
        for i, n in enumerate(names):
            img = Image.open(os.path.join(images, f"{n}.jpg")).convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)   # transforms.Resize's default for PIL
            if camera is not None:
                img = camera(img)
            self.data[i] = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def batches(self, names, batch):
        idx = [self.index[n] for n in names]
        for s in range(0, len(idx), batch):
            x = self.data[idx[s:s + batch]].float() / 255.0
            yield ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last), None


DEVICE = torch.device("cpu")   # set in main(): a GPU when there is one (Kaggle), else the CPU (cloud)


@torch.no_grad()
def plain_scores(model, cache, names, batch):
    model.eval()
    return np.concatenate([model(x.to(DEVICE)).float().cpu().numpy()
                           for x, _ in cache.batches(names, batch)]).astype(np.float64)


def adabn_copy(model, cache, names, batch):
    """M1: a copy of the model whose BatchNorm statistics were re-measured on `names`."""
    adapted = copy.deepcopy(model)
    adapt_batchnorm(adapted, list(cache.batches(names, batch)), DEVICE, False)   # fp32, no autocast
    return adapted


def judge(sick, score, cut):
    sens, spec = sens_spec(sick, score >= cut)
    return {"sens": sens, "spec": spec, "auc": auc(sick, score), "referred": float((score >= cut).mean()),
            "mean_score": float(score.mean())}


def draw_setup(pool, rate, n, rng):
    """A set-up batch of n pool eyes with round(rate * n) sick ones, drawn without replacement."""
    n_sick = int(round(rate * n))
    sick = pool[pool["sick"]]["image"].to_numpy()
    healthy = pool[~pool["sick"]]["image"].to_numpy()
    if n_sick > len(sick) or n - n_sick > len(healthy):
        raise SystemExit(f"The pool is too small for {n} photos at {rate:.0%} sick.")
    return list(rng.choice(sick, n_sick, replace=False)) + list(rng.choice(healthy, n - n_sick, replace=False))


def summarize(draws, m0):
    """Per method x cell: median and 5th-95th percentile over draws, and the shift from M0."""
    rows = []
    for (cond, method, rate), g in draws.groupby(["condition", "method", "rate"]):
        ref = m0["fake" if cond == "fake" else "clean"]
        row = {"condition": cond, "method": method, "rate": rate, "draws": len(g)}
        for k in ("sens", "spec", "auc", "referred"):
            row[k + "_median"] = float(g[k].median())
            row[k + "_p05"], row[k + "_p95"] = (float(v) for v in np.percentile(g[k], [5, 95]))
        row["d_sens_median"] = float((g["sens"] - ref["sens"]).median())
        row["d_spec_median"] = float((g["spec"] - ref["spec"]).median())
        row["share_draws_sens_ge_90"] = float((g["sens"] >= 0.90).mean())
        row["share_draws_sens_ge_85"] = float((g["sens"] >= 0.85).mean())
        rows.append(row)
    return rows


def verdicts(draws, m0, rates, fake_rate):
    """The registered E0 rules (PREREG.md), computed exactly as written there."""
    clean = draws[(draws["condition"] == "clean") & (draws["method"] == "M1")]
    ref = m0["clean"]
    d_sens = clean["sens"] - ref["sens"]
    d_spec = clean["spec"] - ref["spec"]
    med = lambda s, r: float(s[clean["rate"] == r].median())
    out = {}
    if fake_rate in rates:
        a_sens, a_spec = med(d_sens, fake_rate), med(d_spec, fake_rate)
        out["H0a"] = {"rule": f"at p = {fake_rate:.0%}, |median shift| < 2 points for both sensitivity and specificity",
                      "median_d_sens": a_sens, "median_d_spec": a_spec,
                      "pass": abs(a_sens) < 0.02 and abs(a_spec) < 0.02}
    lo, hi = min(rates), max(rates)
    rho_sens = float(spearmanr(clean["rate"], d_sens).correlation)
    rho_spec = float(spearmanr(clean["rate"], d_spec).correlation)
    sens_drop = med(d_sens, lo) - med(d_sens, hi)
    spec_rise = med(d_spec, hi) - med(d_spec, lo)
    out["H0b"] = {"rule": "sensitivity falls and specificity rises with p: Spearman rho(p, shift) < 0 for "
                          "sensitivity and > 0 for specificity over all draws, AND median sensitivity shift at "
                          f"{hi:.0%} at least 2 points below that at {lo:.0%}, AND median specificity shift at "
                          f"{hi:.0%} at least 2 points above that at {lo:.0%}",
                  "rho_sens": rho_sens, "rho_spec": rho_spec, "sens_drop_lo_to_hi": sens_drop,
                  "spec_rise_lo_to_hi": spec_rise,
                  "pass": rho_sens < 0 and rho_spec > 0 and sens_drop >= 0.02 and spec_rise >= 0.02}
    fake = draws[(draws["condition"] == "fake") & (draws["method"] == "M1")]
    if len(fake):
        d0 = abs(m0["fake"]["sens"] - ref["sens"]) + abs(m0["fake"]["spec"] - ref["spec"])
        d1 = float(((fake["sens"] - ref["sens"]).abs() + (fake["spec"] - ref["spec"]).abs()).median())
        out["H0c"] = {"rule": "fake camera: median (|sens shift| + |spec shift|) from the clean operating point "
                              "with AdaBN <= half of it without; not testable if the fake camera moves the "
                              "operating point by less than 5 points",
                      "shift_without_adabn": d0, "median_shift_with_adabn": d1,
                      "pass": None if d0 < 0.05 else d1 <= 0.5 * d0}
    low = [r for r in rates if r <= 0.20]
    worst = min(float(clean[clean["rate"] == r]["sens"].median()) for r in low) if low else None
    out["stop_rule"] = {"rule": "plain AdaBN is called UNSAFE if its median sensitivity at any p <= 20% is below 85%",
                        "lowest_median_sens_at_p_le_20": worst, "unsafe": None if worst is None else worst < 0.85}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="a finished training run (best.pt + val_predictions.csv)")
    p.add_argument("--images", required=True, help="the run's preprocessed training-photo folder")
    p.add_argument("--out", required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.60])
    p.add_argument("--n-setup", type=int, default=200)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--n-healthy", type=int, default=1000, help="healthy eyes kept in the scored clinic")
    p.add_argument("--fake-rate", type=float, default=0.20, help="disease rate of the fake-camera check (<0: skip)")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    args = p.parse_args()
    global DEVICE
    DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "e0_results.json")):
        raise SystemExit(f"{args.out} already holds finished results; use a new --out.")

    settings = {k: v for k, v in vars(args).items() if k not in ("threads", "device", "batch")}
    settings.update(fake_camera={"gamma": FAKE_GAMMA, "blur_sigma": FAKE_BLUR_SIGMA, "jpeg_quality": FAKE_JPEG_QUALITY},
                    view="plain", precision="fp32")
    settings_path = os.path.join(args.out, "settings.json")
    if os.path.exists(settings_path):                     # resuming: the settings must not change
        with open(settings_path) as f:
            old = json.load(f)
        if {k: old.get(k) for k in settings if k != "selfcheck"} != {k: settings[k] for k in settings if k != "selfcheck"}:
            raise SystemExit(f"{settings_path} holds different settings; use a new --out to change them.")

    # ---- the run, its validation eyes, and the two halves
    val = load_predictions(args.run, "val")               # image, patient, level, score (GPU), referred
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"), dropout=arch.get("dropout", 0.2))
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE, memory_format=torch.channels_last).eval()
    print(f"Device: {DEVICE}")
    size = ckpt["size"]
    val["half"] = two_halves(val["patient"].to_numpy(), args.seed)
    pool = val[val["half"] == 0].reset_index(drop=True)
    clinic = val[val["half"] == 1]
    for r in args.rates + ([args.fake_rate] if args.fake_rate >= 0 else []):   # fail before any work
        draw_setup(pool, r, args.n_setup, np.random.default_rng(0))
    rng = np.random.default_rng(args.seed)
    healthy = clinic[~clinic["sick"]]
    keep = healthy.iloc[np.sort(rng.choice(len(healthy), min(args.n_healthy, len(healthy)), replace=False))]
    clinic = pd.concat([clinic[clinic["sick"]], keep]).sort_values("image").reset_index(drop=True)
    if clinic["sick"].nunique() < 2:
        raise SystemExit("The scored clinic needs both sick and healthy eyes; this validation set is too small.")
    print(f"Validation: {len(val)} eyes. Set-up pool (half 0): {len(pool)} eyes, {int(pool['sick'].sum())} sick. "
          f"Scored clinic (half 1): {len(clinic)} eyes, {int(clinic['sick'].sum())} sick + {len(keep)} healthy.")

    t0 = time.time()
    cache = PhotoCache(val["image"].tolist(), args.images, size, ckpt["mean"], ckpt["std"])
    print(f"Decoded {len(val)} photos in {time.time() - t0:.0f} s.")

    # ---- self-check + the frozen cutoff (all validation eyes, this CPU's plain scores)
    t0 = time.time()
    cpu = plain_scores(model, cache, val["image"].tolist(), args.batch)
    rate = len(val) / (time.time() - t0)
    auc_cpu, auc_run = auc(val["sick"], cpu), auc(val["sick"], val["score"])
    corr = float(np.corrcoef(cpu, val["score"])[0, 1])
    print(f"Self-check: val AUC here {auc_cpu:.4f} vs run {auc_run:.4f}; score correlation {corr:.5f} "
          f"({rate:.1f} photos/s).")
    if abs(auc_cpu - auc_run) > 0.002 or corr < 0.999:
        raise SystemExit("Self-check FAILED: these are not the run's scores (wrong photos, folder or checkpoint?).")
    cut = pick_cutoff(val["sick"].to_numpy(), cpu, args.target_sensitivity)
    ref_mean, ref_sd = float(cpu.mean()), float(cpu.std())
    settings["selfcheck"] = {"val_auc_cpu": auc_cpu, "val_auc_run": auc_run, "score_correlation": corr,
                             "cutoff_plain_cpu": cut, "ref_mean": ref_mean, "ref_sd": ref_sd,
                             "photos_per_s": rate}
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    by_name = dict(zip(val["image"], cpu))
    names = clinic["image"].tolist()
    sick = clinic["sick"].to_numpy()
    s0 = np.array([by_name[n] for n in names])
    m0 = {"clean": judge(sick, s0, cut)}
    fake_cache = None
    if args.fake_rate >= 0:
        t0 = time.time()
        fake_names = names + pool["image"].tolist()
        fake_cache = PhotoCache(fake_names, args.images, size, ckpt["mean"], ckpt["std"], camera=fake_camera)
        s0_fake = plain_scores(model, fake_cache, names, args.batch)
        m0["fake"] = judge(sick, s0_fake, cut)
        print(f"Fake camera ready in {time.time() - t0:.0f} s.")
    for k, v in m0.items():
        print(f"M0 ({k}): sens {v['sens']:.1%}, spec {v['spec']:.1%}, AUC {v['auc']:.4f}, referred {v['referred']:.1%}")

    # ---- the draws (resumable: each finished draw is appended to draws.csv)
    draws_path = os.path.join(args.out, "draws.csv")
    done = pd.read_csv(draws_path) if os.path.exists(draws_path) else pd.DataFrame(columns=["condition", "rate", "draw"])
    done_keys = set(zip(done["condition"], done["rate"].round(6), done["draw"]))
    cells = [("clean", r) for r in args.rates] + ([("fake", args.fake_rate)] if args.fake_rate >= 0 else [])
    for ci, (cond, r) in enumerate(cells):
        for d in range(args.draws):
            # Each draw's batch depends only on (seed, cell, draw): stopping and resuming changes nothing.
            draw_rng = np.random.default_rng([args.seed, int(round(r * 1000)), d, 1 if cond == "fake" else 0])
            setup = draw_setup(pool, r, args.n_setup, draw_rng)
            if (cond, round(r, 6), d) in done_keys:
                continue
            t0 = time.time()
            c = fake_cache if cond == "fake" else cache
            base = s0_fake if cond == "fake" else s0
            setup_scores = plain_scores(model, c, setup, args.batch)        # M0 scores of the set-up batch
            adapted = adabn_copy(model, c, setup, args.batch)
            s1 = plain_scores(adapted, c, names, args.batch)
            s3 = (base - setup_scores.mean()) / setup_scores.std() * ref_sd + ref_mean
            rows = []
            for method, sc in (("M1", s1), ("M3", s3)):
                rows.append(dict(condition=cond, rate=r, draw=d, method=method, **judge(sick, sc, cut)))
            pd.DataFrame(rows).to_csv(draws_path, mode="a", header=not os.path.exists(draws_path), index=False)
            print(f"[{cond} p={r:.0%} draw {d + 1}/{args.draws}] M1 sens {rows[0]['sens']:.1%} spec "
                  f"{rows[0]['spec']:.1%} | M3 sens {rows[1]['sens']:.1%} spec {rows[1]['spec']:.1%} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    draws = pd.read_csv(draws_path)
    results = {"m0": m0, "cutoff_plain_cpu": cut, "cells": summarize(draws, m0),
               "verdicts": verdicts(draws, m0, args.rates, args.fake_rate),
               "pool": {"eyes": len(pool), "sick": int(pool["sick"].sum())},
               "clinic": {"eyes": len(clinic), "sick": int(sick.sum())}}
    with open(os.path.join(args.out, "e0_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nVerdicts (rules in PREREG.md):")
    for k, v in results["verdicts"].items():
        print(f"  {k}: {json.dumps({x: y for x, y in v.items() if x != 'rule'})}")


if __name__ == "__main__":
    main()
