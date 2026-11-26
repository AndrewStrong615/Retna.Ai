"""
The screener as one tool: set it up at a new clinic with UNGRADED photos, then give every
photo one of four answers:

    cannot_read    the photo does not look like one the model can read (quality gate)
    clear          below the cutoff, and not close to it
    refer          above the cutoff
    unsure_refer   close to the cutoff, or the 8 views of the photo disagree: send to a person

Two commands:

  setup   (once per clinic/camera)  --tta-run runs\\exp15_tta --images <a few hundred ungraded
          clinic photos, preprocessed> --out runs\\clinic_X
     1. AdaBN: a copy of the model re-measures its BatchNorm statistics on the photos
        (as adapt_site.py; no weight trained, no grade read).
     2. Quality reset: the "does this look like a photo I know?" reference of quality_check.py
        is re-measured on the clinic's own photos, with the ADAPTED model: half the photos
        give the feature cloud, the other half set the distance limit (--quality-quantile).
        Without this, a new camera's photos all look unusual (plan hypothesis H5c). The picture
        limits (eye area, detail) come from the run's EyePACS quality_ref.pt when it exists
        (python quality_check.py fit ...), else from the clinic photos.
     3. Everything the system chose on EyePACS VALIDATION is copied, unchanged: the score
        scale, the 90%-sensitivity cutoff, and the unsure band (the band around the cutoff
        that holds --unsure-share of validation eyes).
  screen  --site runs\\clinic_X --images <photos> --out answers.csv

What this is NOT: a measurement. No grades are read. How well each part works is measured by
site_study.py (set-up methods, unsure readouts) and, once, on a sealed clinic (plan step S5).
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch

from adapt_site import eval_size_of, load_model, loader_for, photos_in
from analyze import validation_choices
from data import make_transforms
from external_test import adapt_batchnorm, check_recipe, source_run
from quality_check import distance, features, fit_cloud, flags, picture_measures
from screening import unsure_band, unsure_flags
from train import write_json
from tta import eight_views


@torch.no_grad()
def views_and_features(model, loader, device, use_amp):
    """Per photo: the 8 TTA view scores (8, n) and the 512 pooled features (plain view)."""
    model.eval()
    all_views, feats = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        vs = []
        for v in eight_views(x):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                vs.append(model(v.contiguous()).float().cpu())
        all_views.append(torch.stack(vs))
        h = model.stages(model.stem(x))
        feats.append(model.pool(h).flatten(1).float().cpu())
    return torch.cat(all_views, dim=1).double().numpy(), torch.cat(feats).double().numpy()


def setup(args):
    site_path = os.path.join(args.out, "site.json")
    if os.path.exists(site_path):
        raise SystemExit(f"{site_path} exists: this clinic was already set up. Use a new --out.")
    photos = photos_in(args.images)
    if len(photos) < args.min_photos:
        raise SystemExit(f"Only {len(photos)} photos; need at least {args.min_photos} (--min-photos).")
    choices = validation_choices([args.tta_run], False, False, args.target_sensitivity)
    (mean, sd), = choices["stats"]
    val_std = choices["combined"]                       # standardized validation TTA scores
    cut = float(choices["cut"])
    band = unsure_band(val_std, cut, args.unsure_share)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run = source_run(args.tta_run)
    check_recipe(run, args.images)
    ckpt = torch.load(os.path.join(run, "best.pt"), map_location=device)
    model = load_model(ckpt, device)
    size = eval_size_of(args.tta_run, ckpt)
    _, tf = make_transforms(size, ckpt["mean"], ckpt["std"])
    adapted = copy.deepcopy(model)
    adapt_batchnorm(adapted, loader_for(photos, args.images, tf, args, use_amp), device, use_amp)
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(ckpt, model={k: v.cpu() for k, v in adapted.state_dict().items()},
                    adapted={"from": os.path.abspath(run), "n_photos": len(photos)}),
               os.path.join(args.out, "model.pt"))

    # Quality reset: feature cloud from one half of the set-up photos, limit from the other half.
    order = np.random.default_rng(args.seed).permutation(len(photos))
    half_a, half_b = photos.iloc[np.sort(order[: len(order) // 2])], photos.iloc[np.sort(order[len(order) // 2:])]
    f_a = features(adapted, loader_for(half_a, args.images, tf, args, use_amp), device)
    f_b = features(adapted, loader_for(half_b, args.images, tf, args, use_amp), device)
    c_mean, c_prec = fit_cloud(f_a)
    q = args.quality_quantile
    limits = {"distance_max": float(np.quantile(distance(f_b, c_mean, c_prec), 1 - q))}
    eyepacs_ref = os.path.join(run, "quality_ref.pt")
    if os.path.exists(eyepacs_ref):
        lim0 = torch.load(eyepacs_ref, map_location="cpu")["limits"]
        limits.update({k: lim0[k] for k in ("eye_area_min", "eye_area_max", "detail_min")})
        picture_source = eyepacs_ref
    else:
        pics = np.array([picture_measures(os.path.join(args.images, f"{n}.jpg")) for n in photos["image"]])
        limits.update({"eye_area_min": float(np.quantile(pics[:, 0], q / 2)),
                       "eye_area_max": float(np.quantile(pics[:, 0], 1 - q / 2)),
                       "detail_min": float(np.quantile(pics[:, 1], q))})
        picture_source = "the set-up photos (no quality_ref.pt in the run)"
    torch.save({"mean": torch.from_numpy(c_mean), "precision": torch.from_numpy(c_prec), "limits": limits},
               os.path.join(args.out, "quality_site.pt"))
    photos[["image"]].to_csv(os.path.join(args.out, "setup_photos.csv"), index=False)
    write_json(site_path, {"tta_run": os.path.abspath(args.tta_run), "run": os.path.abspath(run),
                           "eval_size": int(size), "n_setup": len(photos), "scale": [float(mean), float(sd)],
                           "cutoff": cut, "unsure_band": band, "unsure_share": args.unsure_share,
                           "quality_limits": limits, "picture_limits_from": picture_source,
                           "target_sensitivity": args.target_sensitivity})
    print(f"Set up on {len(photos)} ungraded photos: AdaBN done, quality reference re-measured. Cutoff "
          f"{cut:+.4f} and unsure band +-{band:.4f} (standardized score), both from EyePACS validation.")


def screen(args):
    with open(os.path.join(args.site, "site.json")) as f:
        site = json.load(f)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; choose a new file name.")
    check_recipe(site["run"], args.images)
    table = photos_in(args.images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(os.path.join(args.site, "model.pt"), map_location=device)
    model = load_model(ckpt, device)
    _, tf = make_transforms(site["eval_size"], ckpt["mean"], ckpt["std"])
    views, feats = views_and_features(model, loader_for(table, args.images, tf, args, use_amp), device, use_amp)
    mean, sd = site["scale"]
    views = (views - mean) / sd                          # each view on the validation scale
    score = views.mean(axis=0)
    cut, band = site["cutoff"], site["unsure_band"]
    q = torch.load(os.path.join(args.site, "quality_site.pt"), map_location="cpu")
    pics = np.array([picture_measures(os.path.join(args.images, f"{n}.jpg")) for n in table["image"]])
    dist = distance(feats, q["mean"].numpy(), q["precision"].numpy())
    fl = flags(dist, pics[:, 0], pics[:, 1], q["limits"])
    readable = ~fl.any(axis=1)
    unsure = unsure_flags(score, cut, band, views)
    answer = np.where(~readable, "cannot_read",
                      np.where(unsure, "unsure_refer", np.where(score >= cut, "refer", "clear")))
    out = table[["image"]].assign(answer=answer, score=score, views_agree=((views >= cut).all(axis=0) | (views < cut).all(axis=0)),
                                  unusual_features=fl[:, 0], unusual_eye_area=fl[:, 1], low_detail=fl[:, 2])
    out.to_csv(args.out, index=False)
    counts = pd.Series(answer).value_counts().reindex(["clear", "refer", "unsure_refer", "cannot_read"], fill_value=0)
    print(f"{len(out)} photos: " + ", ".join(f"{k} {v} ({v / len(out):.0%})" for k, v in counts.items())
          + f". Wrote {args.out}.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("setup")
    a.add_argument("--tta-run", required=True, help="the model's <run>_tta folder (cutoff on validation TTA scores)")
    a.add_argument("--images", required=True, help="ungraded set-up photos from the clinic, preprocessed")
    a.add_argument("--out", required=True)
    a.add_argument("--min-photos", type=int, default=200)
    a.add_argument("--unsure-share", type=float, default=0.10)
    a.add_argument("--quality-quantile", type=float, default=0.01)
    a.add_argument("--target-sensitivity", type=float, default=0.90)
    a.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("screen")
    s.add_argument("--site", required=True)
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True)
    for q in (a, s):
        q.add_argument("--batch", type=int, default=32)
        q.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    setup(args) if args.cmd == "setup" else screen(args)


if __name__ == "__main__":
    main()
