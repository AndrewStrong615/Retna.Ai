"""
Get a frozen system ready for a NEW clinic with AdaBN, then screen that clinic's photos.

Why: on every external set the system RANKED eyes well (AUC 0.95-0.97), but its scores ran
high on new cameras, so the cutoff fixed on EyePACS validation referred far too many healthy
eyes (IDRiD: specificity 39.9%). The IDRiD check (S15 vs S15a) showed that most of that shift
sits in the BatchNorm statistics: re-measuring them on photos from the new site (AdaBN, see
external_test.adapt_batchnorm) brought specificity to 75.1% at 96.3% sensitivity, with no
grade used and no weight trained. This script turns that check into a deployment step:

  1. adapt: a few hundred UNGRADED photos from the new clinic -> each model of the system
     gets a copy whose BatchNorm statistics were measured on them. Saved in --out together
     with everything the system chose on EyePACS validation (score scales and both cutoffs),
     which stay exactly as they were.
         python adapt_site.py adapt --tta-runs runs\\exp15_tta
                --images %USERPROFILE%\\retina-data\\clinicX\\calibration_512_enhanced_tight
                --out runs\\site_clinicX
  2. score: screen new photos from that clinic with the adapted copies (8-view TTA, the same
     final score as external_test.py, both cutoffs). Writes one CSV per call.
         python adapt_site.py score --site runs\\site_clinicX
                --images %USERPROFILE%\\retina-data\\clinicX\\week1_512_enhanced_tight
                --out runs\\site_clinicX\\week1.csv

Both image folders must be preprocessed exactly like the system's training photos
(preprocess.py --size 512 --enhance --tight-mask); this is checked with preprocess.json.

What this is NOT: a way to measure accuracy. No grades are read here, so nothing is chosen
and no result is reported. To MEASURE whether AdaBN works on a new site, use
external_test.py --adapt-bn on a graded set (it keeps calibration and scored photos apart).
In real use the calibration photos and the screened photos come from the same clinic, and
scoring a calibration photo is fine: its grade was never used.

Rules it keeps:
  - no weight changes; only the 20 (ResNet-18) BatchNorm layers' running mean/variance;
  - the cutoffs are the ones fixed on EyePACS validation, never re-chosen on the new site;
  - an --out folder is written once (refuses to overwrite), so a site's calibration is a
    fixed, recorded thing: calibrate again into a new folder if the camera changes;
  - fewer than --min-photos (default 200) calibration photos is refused: the statistics of
    a handful of photos are noisy (the IDRiD check used ~258 per half).
No both-eyes blend: a new clinic's files usually carry no patient pairing, and the cutoffs
are then chosen on unblended validation scores (exactly as external_test.py does without --pairs).
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze import validation_choices
from data import FundusDataset, make_transforms
from external_test import adapt_batchnorm, cautious_cutoff, check_recipe, source_run
from model import build_model
from tta import score_eyes
from train import write_json


def photos_in(folder):
    """Every preprocessed photo in a folder, as a table FundusDataset can read (grade unknown = 0)."""
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(".jpg"))
    if not names:
        raise SystemExit(f"No .jpg photos in {folder}. Preprocess the photos first (preprocess.py).")
    return pd.DataFrame({"image": names, "level": 0})


def load_model(ckpt, device):
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def eval_size_of(tta_dir, ckpt):
    """The size the TTA folder chose its cutoff at (tta.py --eval-size); the training size before that option."""
    info_path = os.path.join(tta_dir, "tta.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f).get("eval_size", ckpt["size"])
    return ckpt["size"]


def loader_for(table, images, tf, args, use_amp):
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    return DataLoader(FundusDataset(table, images, tf), batch_size=args.batch,
                      num_workers=workers, pin_memory=use_amp)


def adapt(args):
    site_path = os.path.join(args.out, "site.json")
    if os.path.exists(site_path):
        raise SystemExit(f"{site_path} exists: this site was already calibrated. Use a new --out folder "
                         "(e.g. when the clinic's camera changes), so every calibration stays on record.")
    calib = photos_in(args.images)
    if len(calib) < args.min_photos:
        raise SystemExit(f"Only {len(calib)} calibration photos in {args.images}; need at least {args.min_photos} "
                         "for stable BatchNorm statistics (--min-photos to change).")

    # Everything the system chose, from EyePACS VALIDATION only (as external_test.py without --pairs).
    choices = validation_choices(args.tta_runs, len(args.tta_runs) > 1, False, args.target_sensitivity)
    val = choices["val"]
    cut_caution = cautious_cutoff(val["sick"].to_numpy(), choices["combined"], val["patient"],
                                  args.target_sensitivity, args.n_boot, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(args.out, exist_ok=True)
    models = []
    for i, tta_dir in enumerate(args.tta_runs):
        run = source_run(tta_dir)
        check_recipe(run, args.images)
        ckpt = torch.load(os.path.join(run, "best.pt"), map_location=device)
        model = load_model(ckpt, device)
        size = eval_size_of(tta_dir, ckpt)
        _, eval_tf = make_transforms(size, ckpt["mean"], ckpt["std"])     # the run's own normalization
        adapted = copy.deepcopy(model)
        n_bn = adapt_batchnorm(adapted, loader_for(calib, args.images, eval_tf, args, use_amp), device, use_amp)
        # How far the statistics moved: the mean shift of the first BatchNorm layer, in units of its
        # EyePACS spread. A rough sign of how different the new camera looks to the network.
        bn_old = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        bn_new = next(m for m in adapted.modules() if isinstance(m, torch.nn.BatchNorm2d))
        shift = ((bn_new.running_mean - bn_old.running_mean).abs() / bn_old.running_var.sqrt()).mean().item()
        name = f"model_{i}.pt"
        # Same layout as best.pt (so heatmap-style tools can load it), plus where it came from.
        torch.save(dict(ckpt, model={k: v.cpu() for k, v in adapted.state_dict().items()},
                        adapted={"from": os.path.abspath(run), "images": os.path.abspath(args.images),
                                 "n_photos": len(calib), "eval_size": int(size)}),
                   os.path.join(args.out, name))
        models.append({"tta_run": os.path.abspath(tta_dir), "run": os.path.abspath(run), "file": name,
                       "eval_size": size, "bn_layers": n_bn, "first_bn_mean_shift_sd": shift})
        print(f"  {run}: {n_bn} BatchNorm layers re-measured on {len(calib)} photos "
              f"(first layer moved {shift:.2f} EyePACS spreads on average) -> {name}")

    pd.DataFrame({"image": calib["image"]}).to_csv(os.path.join(args.out, "calibration_photos.csv"), index=False)
    write_json(site_path, {
        "models": models, "calibration_images": os.path.abspath(args.images), "n_calibration": len(calib),
        # The system's choices, fixed on EyePACS validation: how each model's score is standardized,
        # then the two cutoffs on the averaged score.
        "scales": [[float(m), float(s)] for m, s in choices["stats"]],
        "cutoff_standard": float(choices["cut"]), "cutoff_cautious": cut_caution,
        "target_sensitivity": args.target_sensitivity, "both_eyes": False})
    print(f"\nWrote {site_path}. Cutoffs (fixed on EyePACS validation): standard {choices['cut']:+.4f}, "
          f"cautious {cut_caution:+.4f}.\nScreen photos with:\n"
          f"  python adapt_site.py score --site {args.out} --images <folder> --out <file.csv>")


def score(args):
    with open(os.path.join(args.site, "site.json")) as f:
        site = json.load(f)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; choose a new file name.")
    table = photos_in(args.images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    standardized = []
    for m, (mean, sd) in zip(site["models"], site["scales"]):
        check_recipe(m["run"], args.images)
        ckpt = torch.load(os.path.join(args.site, m["file"]), map_location=device)
        model = load_model(ckpt, device)
        _, eval_tf = make_transforms(m["eval_size"], ckpt["mean"], ckpt["std"])
        _, tta = score_eyes(model, loader_for(table, args.images, eval_tf, args, use_amp), device, use_amp)
        standardized.append((tta - mean) / sd)            # same scale the cutoffs were chosen on
    final = np.mean(standardized, axis=0)
    calibrated = set(pd.read_csv(os.path.join(args.site, "calibration_photos.csv"))["image"].astype(str))
    out = table[["image"]].assign(score=final, referred_standard=final >= site["cutoff_standard"],
                                  referred_cautious=final >= site["cutoff_cautious"],
                                  in_calibration=table["image"].isin(calibrated))
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} photos with {len(site['models'])} adapted model(s) x 8 views. "
          f"Referred: {out['referred_standard'].mean():.1%} (standard cutoff), "
          f"{out['referred_cautious'].mean():.1%} (cautious). Wrote {args.out}.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adapt", help="calibrate BatchNorm statistics on a new clinic's ungraded photos")
    a.add_argument("--tta-runs", nargs="+", required=True, help="the frozen system's <run>_tta folders")
    a.add_argument("--images", required=True, help="ungraded calibration photos from the new clinic, preprocessed")
    a.add_argument("--out", required=True, help="new folder for this site, e.g. runs/site_clinicX")
    a.add_argument("--min-photos", type=int, default=200)
    a.add_argument("--target-sensitivity", type=float, default=0.90)
    a.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the cautious cutoff")
    a.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score", help="screen photos from a calibrated site")
    s.add_argument("--site", required=True, help="a folder written by `adapt`")
    s.add_argument("--images", required=True, help="the photos to screen, preprocessed")
    s.add_argument("--out", required=True, help="CSV to write")
    for q in (a, s):
        q.add_argument("--batch", type=int, default=32)
        q.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    adapt(args) if args.cmd == "adapt" else score(args)


if __name__ == "__main__":
    main()
