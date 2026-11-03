"""
Step 4: Train the from-scratch network to detect referable diabetic retinopathy.

Quick sanity check on a small subset first (catches bugs in minutes, not hours):
  python train.py --labels trainLabels.csv --images processed_384 --limit 2000 --epochs 2

Full run:
  python train.py --labels trainLabels.csv --images processed_384 --out runs/first_try
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data import (FundusDataset, compute_mean_std, load_labels,
                  make_transforms, split_by_patient)
from model import RetinaResNet


def safe_auc(labels, probs):
    """AUC: the chance a random sick eye scores higher than a random healthy one.
    0.5 is coin-flipping, 1.0 is perfect."""
    return roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan")


def pick_threshold(labels, probs, target_sensitivity):
    """
    The model outputs a probability, and we need a cutoff above which we say
    "refer to a specialist". In screening, missing a sick eye is much worse
    than a false alarm, so we pick the cutoff that catches at least
    `target_sensitivity` of the sick eyes in the validation set.
    """
    _, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.where(tpr >= target_sensitivity)[0][0]])


def sens_spec(labels, probs, threshold):
    """Sensitivity: share of sick eyes caught. Specificity: share of healthy eyes correctly cleared."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(probs) >= threshold
    sens = (pred & labels).sum() / max(labels.sum(), 1)
    spec = (~pred & ~labels).sum() / max((~labels).sum(), 1)
    return float(sens), float(spec)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()                                  # turns off dropout, freezes BatchNorm statistics
    probs, labels, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        total_loss += loss_fn(logits.float(), y).item() * len(y)
        n += len(y)
        probs.append(torch.sigmoid(logits.float()).cpu())   # sigmoid turns a score into a 0-1 probability
        labels.append(y.cpu())
    return total_loss / max(n, 1), torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="CSV with columns image,level")
    p.add_argument("--images", required=True, help="folder of preprocessed .jpg images")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--limit", type=int, default=0, help="use only N images, for quick tests")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"   # mixed precision: faster, less memory, same results on GPU
    print(f"Device: {device}")

    # ---------------- Data ----------------
    df = load_labels(args.labels)
    exists = df["image"].map(lambda n: os.path.exists(f"{args.images}/{n}.jpg"))
    if (~exists).sum():
        print(f"Skipping {(~exists).sum()} labels with no image file")
    df = df[exists]
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)

    train_df, val_df, test_df = split_by_patient(df, seed=args.seed)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(f"{args.out}/{name}_split.csv", index=False)
        print(f"{name:5s}: {len(part):6d} images, {part['label'].mean():.1%} referable")

    mean, std = compute_mean_std(train_df, args.images)
    train_tf, eval_tf = make_transforms(args.size, mean, std)
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=use_amp)
    train_loader = DataLoader(FundusDataset(train_df, args.images, train_tf),
                              shuffle=True, drop_last=len(train_df) > args.batch, **kw)
    val_loader = DataLoader(FundusDataset(val_df, args.images, eval_tf), **kw)
    test_loader = DataLoader(FundusDataset(test_df, args.images, eval_tf), **kw)

    # ---------------- Model, loss, optimizer ----------------
    model = RetinaResNet().to(device)

    # Most eyes are healthy. Without a correction, the model can look accurate
    # by always guessing "healthy". pos_weight makes the sick eyes, as a group,
    # count as much as the healthy eyes as a group.
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1), device=device))

    # Weight decay gently pulls weights toward zero, which reduces overfitting.
    # It's applied to the conv/linear weights, not to BatchNorm or bias terms.
    decay = [q for q in model.parameters() if q.ndim >= 2]
    no_decay = [q for q in model.parameters() if q.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    # Learning rate schedule: ramp up gently at the start (random weights are
    # fragile), then smoothly decrease so the model settles into a good solution.
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup = max(1, min(len(train_loader), total_steps // 5))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------- Training loop ----------------
    best_auc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)                          # 1. forward: make predictions
            loss = loss_fn(logits.float(), y)              # 2. measure how wrong they are
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()                  # 3. backward: compute gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stop rare huge updates
            scaler.step(optimizer)                         # 4. nudge every weight downhill
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        val_loss, val_probs, val_labels = evaluate(model, val_loader, loss_fn, device, use_amp)
        val_auc = safe_auc(val_labels, val_probs)
        row = dict(epoch=epoch, train_loss=running / max(seen, 1), val_loss=val_loss,
                   val_auc=val_auc, minutes=(time.time() - t0) / 60)
        msg = (f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} | "
               f"val loss {val_loss:.4f} | val AUC {val_auc:.4f}")
        if not math.isnan(val_auc):
            thr = pick_threshold(val_labels, val_probs, args.target_sensitivity)
            sens, spec = sens_spec(val_labels, val_probs, thr)
            row["val_specificity"] = spec
            msg += f" | specificity at {sens:.0%} sensitivity: {spec:.1%}"
        print(msg + f" | {row['minutes']:.1f} min")
        history.append(row)

        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "size": args.size, "threshold": thr, "epoch": epoch},
                       f"{args.out}/best.pt")
            print("    new best, checkpoint saved")
        with open(f"{args.out}/history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ---------------- Final test ----------------
    # The test set is touched exactly once, at the very end, using the model and
    # cutoff chosen on validation data. Peeking at it earlier would make the
    # final number look better than reality.
    if not os.path.exists(f"{args.out}/best.pt"):
        print("No checkpoint was saved (validation set needs both classes). Use more data.")
        return
    ckpt = torch.load(f"{args.out}/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, test_probs, test_labels = evaluate(model, test_loader, loss_fn, device, use_amp)
    test_auc = safe_auc(test_labels, test_probs)
    sens, spec = sens_spec(test_labels, test_probs, ckpt["threshold"])
    print(f"\nTEST (best epoch {ckpt['epoch']}): AUC {test_auc:.4f} | "
          f"sensitivity {sens:.1%} | specificity {spec:.1%}")
    with open(f"{args.out}/test_results.json", "w") as f:
        json.dump({"auc": test_auc, "sensitivity": sens, "specificity": spec,
                   "threshold": ckpt["threshold"], "best_epoch": ckpt["epoch"]}, f, indent=2)


if __name__ == "__main__":
    main()
