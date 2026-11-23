"""
Re-score the VALIDATION eyes with an old run's best.pt, so it can be paired with later runs.

Runs from before the revamp (exp1, exp2) saved no val_predictions.csv, so analyze.py cannot
compare them eye by eye with later runs (paired comparisons are far more sensitive than
comparing two AUCs). This script rebuilds the validation split exactly as train.py does,
scores every validation eye once (plain view, like train.py's own validation pass) and
writes a run-like folder that analyze.py reads unchanged:

    python rescore.py --run runs\\exp2 --labels %USERPROFILE%\\retina-data\\data\\trainLabels.csv
           --images %USERPROFILE%\\retina-data\\processed_384_enhanced
    python analyze.py runs\\exp2_rescored runs\\exp3

Validation only, on purpose: there is no --split option. A test pass belongs to the run's own
training (train.py) or to a frozen system (tta.py --split test), never to a re-scorer.

Old checkpoints: exp1/exp2's best.pt holds model, mean, std, size, threshold and epoch (no
`arch`, `target` or `val_auc`); the network then is ResNet-18 with average pooling, the same
layer names as today's build_model() defaults, so it loads unchanged. Their split was made
AFTER dropping photos without an image file (today's train.py splits first); the two are the
same when no photo is missing (true for the 35,126 EyePACS photos). --split-after-drop
reproduces the old order if some are missing.

Self-check: the rescored AUC should match the run's own best validation AUC (best.pt's
val_auc, or the best epoch in history.json); a GPU run's mixed precision can move it in the
4th decimal. A larger difference means a different split, images or preprocessing.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FundusDataset, load_labels, make_transforms, split_by_patient
from model import build_model
from train import REFERABLE_FROM, pick_threshold, safe_auc, sens_spec, write_json
from tta import score_eyes, write_predictions


def validation_eyes(labels, images, split_seed, after_drop):
    """The run's validation eyes, split exactly as train.py (or the pre-revamp code) did."""
    df = load_labels(labels, referable_from=REFERABLE_FROM)
    exists = df["image"].map(lambda n: os.path.exists(f"{images}/{n}.jpg"))
    if after_drop:                                  # pre-revamp order: drop missing photos, then split
        _, val, _ = split_by_patient(df[exists], seed=split_seed)
    else:                                           # train.py today: split everyone, then drop
        _, val, _ = split_by_patient(df, seed=split_seed)
        val = val[val["image"].map(lambda n: os.path.exists(f"{images}/{n}.jpg"))]
    return val.reset_index(drop=True), int((~exists).sum())


def own_best_auc(run, ckpt):
    """The run's own best validation AUC, from best.pt or history.json (None if neither has it)."""
    if "val_auc" in ckpt:
        return float(ckpt["val_auc"])
    path = os.path.join(run, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
        row = next((r for r in rows if r.get("epoch") == ckpt.get("epoch")), None)
        if row and "val_auc" in row:
            return float(row["val_auc"])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a finished run folder with best.pt, e.g. runs/exp2")
    p.add_argument("--labels", required=True, help="the labels CSV the run trained with")
    p.add_argument("--images", required=True, help="the preprocessed images the run trained on")
    p.add_argument("--out", default=None, help="output folder (default: <run>_rescored)")
    p.add_argument("--split-seed", type=int, default=42, help="the run's split seed (42 for every run so far)")
    p.add_argument("--split-after-drop", action="store_true",
                   help="pre-revamp split order (exp1/exp2): drop photos without an image first, then split")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    out_dir = args.out or args.run.rstrip("/\\") + "_rescored"
    val_out = os.path.join(out_dir, "val_predictions.csv")
    if os.path.exists(val_out):
        raise SystemExit(f"{val_out} exists: this run was already rescored.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})                     # pre-revamp runs: today's defaults are their network
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    _, eval_tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])   # the run's own normalization

    eyes, n_missing = validation_eyes(args.labels, args.images, args.split_seed, args.split_after_drop)
    if n_missing:
        print(f"{n_missing} labels have no image file: the split order matters "
              f"({'old order' if args.split_after_drop else 'train.py order'} used).")
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    loader = DataLoader(FundusDataset(eyes, args.images, eval_tf), batch_size=args.batch,
                        num_workers=workers, pin_memory=use_amp)
    plain, _ = score_eyes(model, loader, device, use_amp)      # view 1 = what train.py scored
    sick = eyes["level"].to_numpy() >= REFERABLE_FROM
    threshold = pick_threshold(sick, plain, args.target_sensitivity)
    os.makedirs(out_dir, exist_ok=True)
    write_predictions(val_out, eyes, plain, threshold)
    a = safe_auc(sick, plain)
    own = own_best_auc(args.run, ckpt)
    sens, spec = sens_spec(sick, plain, threshold)
    write_json(os.path.join(out_dir, "rescore.json"), {
        "run": os.path.abspath(args.run), "images": os.path.abspath(args.images),
        "labels": os.path.abspath(args.labels), "epoch": ckpt.get("epoch"), "split_seed": args.split_seed,
        "split_after_drop": args.split_after_drop, "n_eyes": len(eyes), "val_auc": a, "own_best_val_auc": own,
        "threshold": threshold, "target_sensitivity": args.target_sensitivity})
    print(f"{len(eyes)} validation eyes rescored with {args.run} (epoch {ckpt.get('epoch')}, device {device}): "
          f"AUC {a:.4f}, sensitivity {sens:.1%}, specificity {spec:.1%} at a new 90%-sensitivity cutoff")
    if own is None:
        print("  check: the run recorded no validation AUC to compare with.")
    else:
        verdict = "matches" if abs(a - own) < 0.002 else "DIFFERS: check the split, images and preprocessing"
        print(f"  check: the run's own best val AUC {own:.4f} -> {verdict}")
    print(f"Wrote {val_out}. Pair it with a later run on validation:\n"
          f"  python analyze.py {out_dir} runs\\expN --noise-floor 0.007")


if __name__ == "__main__":
    main()
