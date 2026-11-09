"""
Does a labels file from somewhere else agree with our model? A quick authenticity check.

Before training on labels we did not make ourselves (e.g. testLabels15.csv, the published
grades of the EyePACS 2015 test folder, downloaded from a Kaggle mirror), check that they
belong to the photos: a finished run scores a random sample of those photos, and we measure
the AUC against the file's grades.
  - Genuine labels: AUC close to the run's own validation AUC (~0.9 for exp10).
  - Shuffled, shifted or mismatched labels: AUC near 0.5, i.e. the model's opinion and the
    file have nothing to do with each other. Then do NOT train on them.
Grades from a different grading team or camera would land in between.

This never touches the project's validation or test eyes: it refuses any photo (or patient)
that is in the run's validation or test split.

Usage:
  python check_labels.py --run runs\\exp10 --labels %USERPROFILE%\\retina-data\\data\\testLabels15.csv
         --images %USERPROFILE%\\retina-data\\processed_512_enhanced_tight_unlabeled
  (5,000 random photos by default, ~1 min on the laptop GPU; --limit 0 scores all of them)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FundusDataset, load_labels, make_transforms
from model import build_model
from train import REFERABLE_FROM, safe_auc, sens_spec

GRADE_NAMES = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "proliferative"}


@torch.no_grad()
def score(model, loader, device, use_amp):
    """One plain score per photo (no TTA: this is a check, not a result)."""
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a finished run folder, e.g. runs/exp10")
    p.add_argument("--labels", required=True, help="the labels file to check (columns image,level)")
    p.add_argument("--images", required=True, help="its photos, preprocessed like the run's own")
    p.add_argument("--limit", type=int, default=5000, help="random photos to score (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    labels = load_labels(args.labels, referable_from=REFERABLE_FROM)
    # Keep the project's held-out eyes out of this, even by accident.
    held_names, held_patients = set(), set()
    for split in ("val", "test"):
        path = os.path.join(args.run, f"{split}_split.csv")
        if os.path.exists(path):
            part = pd.read_csv(path, dtype={"image": str, "patient": str})
            held_names |= set(part["image"])
            held_patients |= set(part["patient"])
    clash = labels["image"].isin(held_names) | labels["patient"].isin(held_patients)
    if clash.any():
        raise SystemExit(f"{int(clash.sum())} photos in {args.labels} belong to the run's validation/test "
                         f"patients (e.g. {labels.loc[clash, 'image'].head(3).tolist()}). Not scoring them.")

    exists = labels["image"].map(lambda n: os.path.exists(f"{args.images}/{n}.jpg"))
    print(f"{len(labels)} labels, {int(exists.sum())} with a photo in {args.images}")
    labels = labels[exists]
    if args.limit and len(labels) > args.limit:
        labels = labels.sample(args.limit, random_state=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    _, eval_tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    loader = DataLoader(FundusDataset(labels, args.images, eval_tf), batch_size=args.batch,
                        num_workers=workers, pin_memory=use_amp)
    scores = score(model, loader, device, use_amp)

    sick = labels["level"].to_numpy() >= REFERABLE_FROM
    auc = safe_auc(sick, scores)
    sens, spec = sens_spec(sick, scores, ckpt["threshold"])   # the run's own validation cutoff
    print(f"\nScored {len(labels)} photos with {args.run} (its own val AUC {ckpt['val_auc']:.4f}):")
    print(f"  AUC against the file's grades: {auc:.4f}")
    print(f"  at the run's validation cutoff: sensitivity {sens:.1%}, specificity {spec:.1%}")
    print("  mean score by the file's grade (should climb from 0 to 4):")
    for g, part in pd.Series(scores).groupby(labels["level"].to_numpy()):
        print(f"    grade {g} ({GRADE_NAMES.get(g, '?'):13s}) {part.mean():+.2f}  ({len(part)} photos)")
    gap = ckpt["val_auc"] - auc
    if auc < 0.6:
        verdict = "NO: the grades do not match these photos (shuffled or mismatched). Do not train on them."
    elif gap > 0.05:
        verdict = "PARTLY: clearly related, but much weaker than on our own validation eyes. Look closer."
    else:
        verdict = "YES: the grades agree with the model about as well as our own validation labels do."
    print(f"\nGenuine? {verdict}")


if __name__ == "__main__":
    main()
