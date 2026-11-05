"""
Test-time augmentation (TTA): score every eye 8 times, in 8 orientations, and average.

Why it helps: during training the model saw every photo flipped and rotated at random,
so it should give the same score whichever way an eye is turned. In practice each
orientation gives a slightly different score, partly from noise (a lesion that happens
to fall on the edge of a 32x32 cell in one orientation and in the middle of one in
another). Averaging the 8 scores cancels part of that noise, like asking the same
grader 8 times and taking the mean. No training: the weights stay exactly the same.

The 8 orientations are the 8 symmetries of a square: the photo as is, flipped
left-right, flipped up-down, turned 180 degrees, and the same four after mirroring
across the diagonal (which gives the 90 and 270 degree turns). Because the image is
square, every one of them is EXACT: pixels are only moved, never blended, and no
corner needs filling, unlike the random rotations used in training.

How it fits the project rules:
  1. Validation first (default). Writes <out>/val_predictions.csv with the averaged
     scores and a NEW 90%-sensitivity cutoff picked on these validation scores.
     Compare with the plain run like any other change:
        python analyze.py runs\\exp10 runs\\exp10_tta --noise-floor 0.007
  2. Only after deciding on validation: score the test eyes once.
        python tta.py --run runs\\exp10 --images ... --split test
     Test eyes are scored with the cutoff chosen in step 1. It refuses to run
     twice, and refuses step 1 again once the test eyes have been scored.
The output folder looks like a run to analyze.py, so --both-eyes, --ensemble and
--final-test work on it unchanged.

Usage:
  python tta.py --run runs\\exp10 --images %USERPROFILE%\\retina-data\\processed_512_enhanced_tight
  (writes runs\\exp10_tta; ~2-3 min for the 5,268 validation eyes on the laptop GPU)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FundusDataset, make_transforms
from model import build_model
from train import REFERABLE_FROM, pick_threshold, safe_auc, sens_spec, write_json


def eight_views(x):
    """
    The 8 symmetries of a square, for a batch x of shape (batch, 3, H, W) with H == W.
    flip(3) mirrors left-right, flip(2) up-down, transpose(2, 3) mirrors across the
    diagonal. The first view is the photo unchanged.
    """
    d = x.transpose(2, 3)
    return [x, x.flip(3), x.flip(2), x.flip(2, 3),     # as is, mirrored, upside down, turned 180
            d, d.flip(3), d.flip(2), d.flip(2, 3)]     # the diagonal mirror and the 90/270 turns


@torch.no_grad()
def score_eyes(model, loader, device, use_amp):
    """For every eye: its plain score (view 1 only) and its TTA score (mean of the 8 views)."""
    model.eval()                               # dropout off, BatchNorm uses its saved statistics
    plain, averaged = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        views = []
        for v in eight_views(x):
            # .contiguous(): a flipped or transposed tensor is only a different "view"
            # of the same memory; the convolutions run fastest on a fresh, ordered copy.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                views.append(model(v.contiguous()).float())
        views = torch.stack(views)             # (8, batch)
        plain.append(views[0].cpu())
        averaged.append(views.mean(dim=0).cpu())
    return torch.cat(plain).numpy().astype(np.float64), torch.cat(averaged).numpy().astype(np.float64)


def write_predictions(path, eyes, scores, threshold):
    """Same columns analyze.py reads from a training run's predictions files."""
    out = eyes[["image", "patient", "level"]].copy()
    out["score"] = scores
    out["referred"] = scores >= threshold
    out.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a finished run folder, e.g. runs/exp10")
    p.add_argument("--images", required=True, help="the preprocessed images that run trained on")
    p.add_argument("--out", default=None, help="output folder (default: <run>_tta)")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="val first; test only once, after the decision was made on validation")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=None, help="image-loading helpers (default: up to 8)")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    args = p.parse_args()
    out_dir = args.out or args.run.rstrip("/\\") + "_tta"
    os.makedirs(out_dir, exist_ok=True)
    val_out, test_out = (os.path.join(out_dir, f"{s}_predictions.csv") for s in ("val", "test"))
    info_path = os.path.join(out_dir, "tta.json")

    # The order of the steps protects the test set (see the docstring).
    if args.split == "val" and os.path.exists(test_out):
        raise SystemExit(f"{out_dir} already scored the test eyes. Its validation choices are final; "
                         "rescoring validation now would change the cutoff after the test was seen.")
    if args.split == "test":
        if not os.path.exists(info_path):
            raise SystemExit(f"Run --split val first: the cutoff must be chosen on validation ({info_path} missing).")
        if os.path.exists(test_out):
            raise SystemExit(f"{test_out} exists: the test eyes were already scored once.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    _, eval_tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])   # the run's own normalization

    # The exact eyes (and order) the run scored, read from its own predictions file.
    # Only image/patient/level are used; the run's scores are not.
    source = os.path.join(args.run, f"{args.split}_predictions.csv")
    eyes = pd.read_csv(source, dtype={"image": str, "patient": str})[["image", "patient", "level"]]
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    loader = DataLoader(FundusDataset(eyes, args.images, eval_tf), batch_size=args.batch,
                        num_workers=workers, pin_memory=use_amp)
    print(f"Device: {device} | model from {args.run} (epoch {ckpt['epoch']}) | "
          f"{len(eyes)} {args.split} eyes x 8 orientations")
    start = time.time()
    plain, tta = score_eyes(model, loader, device, use_amp)
    print(f"Scored in {(time.time() - start) / 60:.1f} min")
    sick = eyes["level"].to_numpy() >= REFERABLE_FROM

    if args.split == "val":
        threshold = pick_threshold(sick, tta, args.target_sensitivity)   # chosen on validation only
        write_predictions(val_out, eyes, tta, threshold)
        a_plain, a_tta = safe_auc(sick, plain), safe_auc(sick, tta)
        sens, spec = sens_spec(sick, tta, threshold)
        write_json(info_path, {"run": os.path.abspath(args.run), "images": os.path.abspath(args.images),
                               "views": 8, "threshold": threshold, "val_auc_plain": a_plain,
                               "val_auc_tta": a_tta, "target_sensitivity": args.target_sensitivity})
        # Self-check: view 1 is exactly what train.py scored, so its AUC should match the run's
        # best val AUC (tiny differences on the GPU come from mixed precision).
        print(f"  check: plain view AUC {a_plain:.4f} (run's own best val AUC {ckpt['val_auc']:.4f})")
        print(f"  validation AUC with 8-view TTA: {a_tta:.4f} ({a_tta - a_plain:+.4f}) | "
              f"sensitivity {sens:.1%}, specificity {spec:.1%}")
        print(f"\nWrote {val_out}. Compare on validation:\n"
              f"  python analyze.py {args.run} {out_dir} --noise-floor 0.007")
    else:
        with open(info_path) as f:
            threshold = json.load(f)["threshold"]              # the cutoff chosen on validation
        write_predictions(test_out, eyes, tta, threshold)
        # The numbers are NOT printed here: report them with analyze.py (e.g. --final-test),
        # so the test set is read once, in one place.
        print(f"Wrote {test_out}. Report it once with analyze.py, e.g.\n"
              f"  python analyze.py {out_dir} --both-eyes --final-test")


if __name__ == "__main__":
    main()
