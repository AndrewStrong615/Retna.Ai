"""
Pretend clinics on a REAL other clinic's photos: which label-free set-up puts it back on the
EyePACS operating point, at which disease rate, with how many photos? (E1-E4 of PREREG.md.)

site_sim.py (E0) built pretend clinics from EyePACS validation eyes, so only the disease rate
changed. Here the photos come from another clinic (another camera), e.g. DeepDRiD:

  1. The clinic's patients are split in two halves (both eyes, every photo of a patient, in
     the same half). Half 0 = set-up pool, half 1 = scored clinic.
  2. A pretend clinic = a set-up batch of N photos from the pool with share p sick (grades
     only BUILD the batch; no method reads them), then the scored clinic is judged at the
     cutoff fixed on EyePACS validation.
  3. Repeated --draws times for every (p, N) cell.

Methods (none reads a grade from the clinic, except the yardstick REF):
  M0   nothing: the model and cutoff as shipped (one number, no draws).
  M1   AdaBN: BatchNorm statistics re-measured on the set-up batch.
  M2   blended AdaBN: EyePACS and clinic statistics mixed, w = prior_n / (prior_n + N).
  M3   score reset: set-up batch scores matched to EyePACS validation's mean and spread.
  M4   referral-rate reset: refer the share the clinic expects from its STATED disease rate
       (true p + each --stated-offset), at the EyePACS operating point.
  M5   M1 + an estimate of the batch's disease rate from its scores (EM); judged like M1,
       plus the estimate's error (est_rate - p).
  REF  graded reset (NOT label-free, the yardstick): the 90%-sensitivity cutoff chosen on
       the N set-up photos WITH their grades, network untouched.
Every row also carries the "unsure, refer" readouts: a band around the cutoff holding 10%
of EyePACS validation eyes (plain view here, so no view disagreement).

Reference numbers (cutoff, score mean/spread, Platt fit, the EyePACS operating point) come
from EyePACS VALIDATION only: from this machine's own scores when --val-images is given
(self-check: AUC equal to the run's), else from the run's val_predictions.csv.

Refuses any photo listed in --sealed (DeepDRiD's sealed part is for the final screener run).

Usage:
    python site_study.py --run runs/exp15 --val-images <EyePACS processed_512_enhanced_tight>
        --clinic-labels deepdrid/deepdrid_labels.csv --clinic-images deepdrid/processed_512_enhanced_tight
        --sealed deepdrid/deepdrid_sealed.csv --out runs/e2_deepdrid_exp15
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from analyze import REFERABLE_FROM, auc, load_predictions, pick_cutoff, sens_spec
from external_test import adapt_batchnorm, two_halves
from model import build_model
import site_sim
from site_sim import PhotoCache, adabn_copy, draw_setup, judge, plain_scores
from screening import blend_batchnorm, em_prevalence, platt_fit, rate_cutoff, unsure_band, unsure_flags, unsure_readouts


def load_clinic(labels_csv, sealed_csv, images):
    df = pd.read_csv(labels_csv, dtype={"image": str})
    if not {"image", "level"} <= set(df.columns):
        raise SystemExit(f"{labels_csv} needs columns image,level (patient optional).")
    if "patient" not in df:
        df["patient"] = df["image"].str.split("_").str[0]      # EyePACS / DeepDRiD naming
    if "part" in df:
        df = df[df["part"] != "sealed"]
    if sealed_csv:
        sealed = set(pd.read_csv(sealed_csv, dtype={"image": str})["image"])
        if df["image"].isin(sealed).any():
            raise SystemExit("The clinic table holds sealed photos: they are for the final screener run only.")
    has_photo = df["image"].map(lambda n: os.path.exists(os.path.join(images, f"{n}.jpg")))
    if not has_photo.all():
        print(f"Left out {int((~has_photo).sum())} labels without a preprocessed photo.")
    df = df[has_photo].copy()
    df["sick"] = df["level"] >= REFERABLE_FROM
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--val-images", default=None, help="EyePACS validation photos (for the self-check); optional")
    p.add_argument("--clinic-labels", required=True)
    p.add_argument("--clinic-images", required=True)
    p.add_argument("--sealed", default=None, help="CSV of photos that must never be scored here")
    p.add_argument("--out", required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.60])
    p.add_argument("--n-setup", type=int, nargs="+", default=[200])
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--methods", nargs="+", default=["M1", "M2", "M3", "M4", "M5", "REF"])
    p.add_argument("--prior-n", type=float, default=64, help="M2: what the EyePACS statistics are worth, in photos")
    p.add_argument("--stated-offset", type=float, nargs="+", default=[0.0, -0.05, 0.05], help="M4")
    p.add_argument("--unsure-share", type=float, default=0.10)
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    site_sim.DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
                                   else args.device)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "results.json")):
        raise SystemExit(f"{args.out} already holds finished results; use a new --out.")
    settings = {k: v for k, v in vars(args).items() if k not in ("device", "batch")}
    settings_path = os.path.join(args.out, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            old = json.load(f)
        if {k: old.get(k) for k in settings} != settings:
            raise SystemExit(f"{settings_path} holds different settings; use a new --out to change them.")

    # ---- the model and its EyePACS validation reference
    val = load_predictions(args.run, "val")
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"), dropout=arch.get("dropout", 0.2))
    model.load_state_dict(ckpt["model"])
    model = model.to(site_sim.DEVICE, memory_format=torch.channels_last).eval()
    size, mean, std = ckpt["size"], ckpt["mean"], ckpt["std"]
    if args.val_images:
        vcache = PhotoCache(val["image"].tolist(), args.val_images, size, mean, std)
        ref = plain_scores(model, vcache, val["image"].tolist(), args.batch)
        del vcache
        a_here, a_run = auc(val["sick"], ref), auc(val["sick"], val["score"])
        print(f"Self-check: val AUC here {a_here:.4f} vs run {a_run:.4f}")
        if abs(a_here - a_run) > 0.002:
            raise SystemExit("Self-check FAILED: these are not the run's scores.")
        ref_source = "this machine's scores"
    else:
        ref = val["score"].to_numpy()
        ref_source = "the run's val_predictions.csv (no --val-images)"
    vsick = val["sick"].to_numpy()
    cut = pick_cutoff(vsick, ref, args.target_sensitivity)
    ref_mean, ref_sd = float(ref.mean()), float(ref.std())
    ref_sens, ref_spec = sens_spec(vsick, ref >= cut)
    platt = platt_fit(ref, vsick)
    train_prior = float(vsick.mean())
    band = unsure_band(ref, cut, args.unsure_share)
    reference = {"source": ref_source, "cutoff": cut, "mean": ref_mean, "sd": ref_sd, "sens": ref_sens,
                 "spec": ref_spec, "platt": platt, "val_sick_share": train_prior, "unsure_band": band}
    print(f"EyePACS validation reference ({ref_source}): cutoff {cut:+.4f}, sens {ref_sens:.1%}, spec {ref_spec:.1%}")

    # ---- the clinic: halves by patient
    clinic_all = load_clinic(args.clinic_labels, args.sealed, args.clinic_images)
    clinic_all["half"] = two_halves(clinic_all["patient"].to_numpy(), args.seed)
    pool = clinic_all[clinic_all["half"] == 0].reset_index(drop=True)
    scored = clinic_all[clinic_all["half"] == 1].reset_index(drop=True)
    if scored["sick"].nunique() < 2:
        raise SystemExit("The scored half needs both sick and healthy photos.")
    for r in args.rates:
        for n in args.n_setup:
            draw_setup(pool, r, n, np.random.default_rng(0))       # refuse impossible cells before any work
    print(f"Clinic: {len(clinic_all)} photos, {clinic_all['patient'].nunique()} patients. Pool {len(pool)} photos "
          f"({int(pool['sick'].sum())} sick); scored {len(scored)} photos ({int(scored['sick'].sum())} sick).")
    settings["reference"] = reference
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    cache = PhotoCache(clinic_all["image"].tolist(), args.clinic_images, size, mean, std)
    names, sick = scored["image"].tolist(), scored["sick"].to_numpy()
    s0 = plain_scores(model, cache, names, args.batch)
    m0 = dict(judge(sick, s0, cut), **unsure_readouts(sick, s0, cut, unsure_flags(s0, cut, band)))
    print(f"M0: sens {m0['sens']:.1%}, spec {m0['spec']:.1%}, AUC {m0['auc']:.4f}, referred {m0['referred']:.1%}")
    pool_scores = dict(zip(pool["image"], plain_scores(model, cache, pool["image"].tolist(), args.batch)))
    pool_sick = dict(zip(pool["image"], pool["sick"]))

    draws_path = os.path.join(args.out, "draws.csv")
    done = pd.read_csv(draws_path) if os.path.exists(draws_path) else pd.DataFrame(columns=["rate", "n_setup", "draw"])
    done_keys = set(zip(done["rate"].round(6), done["n_setup"], done["draw"]))

    def row(method, score, rate, n, d, cutoff=cut, **extra):
        flag = unsure_flags(score, cutoff, band)
        return dict(rate=rate, n_setup=n, draw=d, method=method, **judge(sick, score, cutoff),
                    **unsure_readouts(sick, score, cutoff, flag), **extra)

    for n in args.n_setup:
        for r in args.rates:
            for d in range(args.draws):
                rng = np.random.default_rng([args.seed, int(round(r * 1000)), n, d])
                setup = draw_setup(pool, r, n, rng)
                if (round(r, 6), n, d) in done_keys:
                    continue
                t0 = time.time()
                ss = np.array([pool_scores[x] for x in setup])
                rows = []
                adapted = None
                if {"M1", "M2", "M5"} & set(args.methods):
                    adapted = adabn_copy(model, cache, setup, args.batch)
                if {"M1", "M5"} & set(args.methods):
                    s1 = plain_scores(adapted, cache, names, args.batch)
                    if "M1" in args.methods:
                        rows.append(row("M1", s1, r, n, d))
                    if "M5" in args.methods:
                        ss1 = plain_scores(adapted, cache, setup, args.batch)
                        est = em_prevalence(ss1, platt, train_prior)
                        rows.append(row("M5", s1, r, n, d, est_rate=est, est_error=est - r))
                if "M2" in args.methods:
                    blended = blend_batchnorm(copy.deepcopy(adapted), model, n, args.prior_n)
                    rows.append(row("M2", plain_scores(blended, cache, names, args.batch), r, n, d))
                if "M3" in args.methods:
                    rows.append(row("M3", (s0 - ss.mean()) / ss.std() * ref_sd + ref_mean, r, n, d))
                if "M4" in args.methods:
                    for off in args.stated_offset:
                        c4 = rate_cutoff(ss, r + off, ref_spec, args.target_sensitivity)
                        rows.append(row(f"M4{off:+.2f}", s0, r, n, d, cutoff=c4, stated_rate=r + off))
                if "REF" in args.methods:
                    g = np.array([pool_sick[x] for x in setup])
                    c_ref = pick_cutoff(g, ss, args.target_sensitivity) if 0 < g.sum() < len(g) else float("nan")
                    if np.isfinite(c_ref):
                        rows.append(row("REF", s0, r, n, d, cutoff=c_ref))
                pd.DataFrame(rows).to_csv(draws_path, mode="a", header=not os.path.exists(draws_path), index=False)
                print(f"[N={n} p={r:.0%} draw {d + 1}/{args.draws}] "
                      + " | ".join(f"{x['method']} {x['sens']:.0%}/{x['spec']:.0%}" for x in rows)
                      + f" ({time.time() - t0:.0f} s)", flush=True)

    draws = pd.read_csv(draws_path)
    cells = []
    for (method, n, r), g in draws.groupby(["method", "n_setup", "rate"]):
        c = {"method": method, "n_setup": int(n), "rate": float(r), "draws": len(g)}
        for k in ("sens", "spec", "auc", "referred", "unsure_share", "sens_unsure_refer", "spec_unsure_refer"):
            c[k + "_median"] = float(g[k].median())
            c[k + "_p05"], c[k + "_p95"] = (float(v) for v in np.percentile(g[k], [5, 95]))
        c["safe_draw_rate"] = float((g["sens"] >= args.target_sensitivity).mean())
        if "est_error" in g and g["est_error"].notna().any():
            c["est_error_median"] = float(g["est_error"].median())
        cells.append(c)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"reference": reference, "m0": m0, "cells": cells,
                   "pool": {"photos": len(pool), "sick": int(pool["sick"].sum())},
                   "scored": {"photos": len(scored), "sick": int(sick.sum())}}, f, indent=2)
    print(f"Wrote {args.out}/results.json ({len(cells)} cells).")


if __name__ == "__main__":
    main()
