"""
"Is this a photo the model knows how to read?" A check that runs BEFORE a DR score is trusted.

Today every photo gets a DR score, even a blurry one, a photo of the wrong thing, or a
camera the model never saw. This check flags photos that do not look like the photos the
model learned from, with no grades and no new training, in two independent ways:

  1. Feature distance (Lee et al. 2018, "A simple unified framework for detecting
     out-of-distribution samples"): the trained network turns every photo into 512 numbers
     just before its final layer (after pooling). For the training photos those numbers form
     a cloud; we measure its centre and shape (mean and covariance) and, for a new photo,
     how far outside the cloud it lands, in units of the cloud's own spread (Mahalanobis
     distance). A photo of something else, or a very unusual photo, lands far away.
     The covariance of 512 numbers from ~2,000 photos is noisy, so it is "shrunk" a little
     toward a plain sphere (written out in fit_cloud).
  2. Two simple picture measurements on the preprocessed photo (preprocess.py makes the
     background outside the eye flat: gray 128 with --enhance, black without):
       - eye_area: share of the square covered by the eye (not background colour). A failed
         eye detection, a cropped or a non-fundus photo gives an unusual value;
       - detail: average distance from the background colour inside the eye. With --enhance this is the local
         contrast left after removing the smooth colour, so a blurred, dark or washed-out
         photo has little of it.

Where the limits come from: measured on the run's TRAINING photos (the cloud), with the
cutoffs set on the VALIDATION photos as plain pictures (their grades are never read): a
photo is flagged when it is further out than all but `--quantile` (default 1%) of the
validation photos on any of the three measures. So by construction about 1-3% of normal
EyePACS photos are flagged; the question is how many unusual photos are.

Usage:
  python quality_check.py fit --run runs\\exp15 --images %USERPROFILE%\\retina-data\\processed_512_enhanced_tight
         (writes runs\\exp15\\quality_ref.pt; ~2-3 min on the laptop GPU for 2,000 + 2,000 photos)
  python quality_check.py check --run runs\\exp15 --images <preprocessed folder> --out quality.csv

A real check worth doing on the laptop: DDR's grade-5 photos ("ungradable", left out by
convert_ddr.py) should be flagged far more often than DDR's gradable ones. That tells
whether the flags mean "a grader could not read this" and not only "another camera".
Note: every photo of a new camera may land a little outside the cloud (the camera shift seen
on Messidor-2/APTOS/IDRiD); adapt_site.py's re-measured BatchNorm statistics change the
features too, so fit and check with the same, unadapted run.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import FundusDataset, make_transforms
from model import build_model

SHRINK = 0.1           # how far the covariance is pulled toward a sphere (0 = not at all)


@torch.no_grad()
def features(model, loader, device):
    """The 512 numbers per photo that the final layer reads (after pooling, before dropout/head)."""
    model.eval()
    out = []
    for x, _ in loader:
        h = model.stages(model.stem(x.to(device)))
        out.append(model.pool(h).flatten(1).float().cpu())
    return torch.cat(out).double().numpy()


def fit_cloud(f):
    """
    Centre and inverse covariance of the training photos' features.
      cov    = ordinary covariance of the features;
      target = a sphere with the same average spread (trace / dimensions);
      shrunk = (1 - SHRINK) * cov + SHRINK * target.
    Shrinking keeps the rarely-varying directions from getting huge weights just from noise.
    """
    mean = f.mean(axis=0)
    centred = f - mean
    cov = centred.T @ centred / max(len(f) - 1, 1)
    target = np.eye(len(mean)) * np.trace(cov) / len(mean)
    shrunk = (1 - SHRINK) * cov + SHRINK * target
    return mean, np.linalg.inv(shrunk)


def distance(f, mean, precision):
    """Mahalanobis distance: sqrt((f - mean)^T  precision  (f - mean)), one number per photo."""
    d = f - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d).clip(min=0))


def picture_measures(path, tol=6):
    """
    eye_area and detail of one preprocessed photo (see the docstring). The background colour
    is read from the four corners (always outside the eye after preprocessing): gray 128 with
    --enhance, black without it.
    """
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    gray = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    off_gray = np.abs(a - gray).max(axis=2)
    eye = off_gray > tol
    return float(eye.mean()), float(off_gray[eye].mean()) if eye.any() else 0.0


def measure(model, table, images, tf, device, batch, workers):
    loader = DataLoader(FundusDataset(table.assign(level=0), images, tf), batch_size=batch, num_workers=workers)
    feats = features(model, loader, device)
    pics = np.array([picture_measures(os.path.join(images, f"{n}.jpg")) for n in table["image"]])
    return feats, pics[:, 0], pics[:, 1]


def load_run(run, device):
    ckpt = torch.load(os.path.join(run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    _, tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])
    return model, tf


def fit(args, device):
    model, tf = load_run(args.run, device)
    split = lambda s: pd.read_csv(os.path.join(args.run, f"{s}_split.csv"), dtype={"image": str})
    train = split("train").sample(min(args.n, len(split("train"))), random_state=0)
    val = split("val")
    val = val.sample(min(args.n, len(val)), random_state=0)        # pictures only: grades never read
    f_train, _, _ = measure(model, train[["image"]], args.images, tf, device, args.batch, args.workers)
    mean, precision = fit_cloud(f_train)
    f_val, area_val, detail_val = measure(model, val[["image"]], args.images, tf, device, args.batch, args.workers)
    d_val = distance(f_val, mean, precision)
    q = args.quantile
    limits = {"distance_max": float(np.quantile(d_val, 1 - q)),
              "eye_area_min": float(np.quantile(area_val, q / 2)), "eye_area_max": float(np.quantile(area_val, 1 - q / 2)),
              "detail_min": float(np.quantile(detail_val, q))}
    torch.save({"mean": torch.from_numpy(mean), "precision": torch.from_numpy(precision), "limits": limits,
                "n_train": len(train), "n_val": len(val), "quantile": q, "shrink": SHRINK},
               os.path.join(args.run, "quality_ref.pt"))
    flagged = flags(d_val, area_val, detail_val, limits).any(axis=1).mean()
    print(f"Fitted on {len(train)} training photos, limits set on {len(val)} validation photos (no grades):")
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in limits.items()))
    print(f"  validation photos flagged by at least one measure: {flagged:.1%} "
          f"(expected ~{q:.0%}-{3 * q:.0%} by construction)")


def flags(dist, area, detail, lim):
    return np.stack([dist > lim["distance_max"],
                     (area < lim["eye_area_min"]) | (area > lim["eye_area_max"]),
                     detail < lim["detail_min"]], axis=1)


def check(args, device):
    ref = torch.load(os.path.join(args.run, "quality_ref.pt"), map_location="cpu")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; choose a new file name.")
    model, tf = load_run(args.run, device)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(args.images) if f.lower().endswith(".jpg"))
    table = pd.DataFrame({"image": names})
    f, area, detail = measure(model, table, args.images, tf, device, args.batch, args.workers)
    dist = distance(f, ref["mean"].numpy(), ref["precision"].numpy())
    fl = flags(dist, area, detail, ref["limits"])
    out = table.assign(feature_distance=dist, eye_area=area, detail=detail, unusual_features=fl[:, 0],
                       unusual_eye_area=fl[:, 1], low_detail=fl[:, 2], ok=~fl.any(axis=1))
    out.to_csv(args.out, index=False)
    print(f"{len(out)} photos: {out['ok'].mean():.1%} look like the training photos | flagged: "
          f"features {fl[:, 0].mean():.1%}, eye area {fl[:, 1].mean():.1%}, low detail {fl[:, 2].mean():.1%}. "
          f"Wrote {args.out}.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit", help="measure the training photos' cloud and set the limits on validation photos")
    a.add_argument("--n", type=int, default=2000, help="photos per part (train for the cloud, val for the limits)")
    a.add_argument("--quantile", type=float, default=0.01, help="share of validation photos beyond each limit")
    c = sub.add_parser("check", help="flag unusual photos in a folder")
    c.add_argument("--out", required=True, help="CSV to write")
    for q in (a, c):
        q.add_argument("--run", required=True, help="a finished run (fit writes quality_ref.pt into it)")
        q.add_argument("--images", required=True, help="preprocessed photos (fit: the run's own --images)")
        q.add_argument("--batch", type=int, default=32)
        q.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit(args, device) if args.cmd == "fit" else check(args, device)


if __name__ == "__main__":
    main()
