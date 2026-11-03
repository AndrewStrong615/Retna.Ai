"""
Step 4: Train the from-scratch network to detect referable diabetic retinopathy.

Quick sanity check on a small subset first (catches bugs in minutes, not hours):
  python train.py --labels trainLabels.csv --images processed_384_enhanced --limit 2000 --epochs 2 --out runs/sanity

Full run (learns from the full 0-4 grade, the default):
  python train.py --labels trainLabels.csv --images processed_384_enhanced --out runs/exp3

The same run with the old yes/no target, for a fair side-by-side comparison:
  python train.py --labels trainLabels.csv --images processed_384_enhanced --target binary --out runs/exp4

The exp2 recipe (yes/no target, nearest-neighbor rotation, black corners):
  python train.py ... --target binary --rotate-interp nearest --rotate-fill black
(With --workers 0 on a CPU this reproduces the old code's numbers bit for bit.
With loader workers, the random augmentation order differs from epoch 2 on, so
expect the usual run-to-run noise rather than identical numbers.)

Every run writes into its --out folder: config.json (all settings), history.json
(one row per epoch), best.pt (the best model), val_predictions.csv,
test_predictions.csv and test_results.json. A run refuses to write into a folder
that already holds results, unless you pass --overwrite.
"""

import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data import (FundusDataset, compute_mean_std, estimate_background, load_labels,
                  make_transforms, split_by_patient)
from model import RetinaResNet

REFERABLE_FROM = 2          # grade 2 (moderate) or worse = refer to a specialist
NUM_GRADES = 5              # EyePACS grades 0..4
RESULT_FILES = ["config.json", "history.json", "best.pt", "val_predictions.csv",
                "test_predictions.csv", "test_results.json"]


# ---------------------------------------------------------------- small helpers

def code_version():
    """The git commit of this code, e.g. "284aabb", and whether it had uncommitted
    edits. (None, None) outside a git checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=here).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, cwd=here)
        dirty = bool(status.stdout.strip()) if commit and status.returncode == 0 else None
        return commit, dirty
    except OSError:                               # git not installed
        return None, None


def to_py(obj):
    """
    Turn numpy numbers/arrays and 0-d tensors into plain Python floats, ints and
    lists. json.dump refuses numpy types, and torch.load (which by default only
    loads "safe" plain types) refuses numpy numbers inside a checkpoint. Running
    everything through this before saving avoids a crash at the end of a long run.
    """
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return to_py(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(to_py(obj), f, indent=2)


# ---------------------------------------------------------------- metrics

def safe_auc(labels, scores):
    """AUC: the chance a random sick eye scores higher than a random healthy one.
    0.5 is coin-flipping, 1.0 is perfect."""
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def pick_threshold(labels, scores, target_sensitivity):
    """
    The model outputs a score, and we need a cutoff above which we say
    "refer to a specialist". In screening, missing a sick eye is much worse
    than a false alarm, so we pick the cutoff that catches at least
    `target_sensitivity` of the sick eyes in the validation set.

    drop_intermediate=False keeps every possible cutoff. (The default thins out
    points that lie on straight stretches of the ROC curve; that is fine for
    drawing the curve, but can skip the best cutoff when several scores tie.)
    """
    _, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    return float(thresholds[np.where(tpr >= target_sensitivity)[0][0]])


def sens_spec(labels, scores, threshold):
    """Sensitivity: share of sick eyes caught. Specificity: share of healthy eyes correctly cleared."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    sens = (pred & labels).sum() / max(labels.sum(), 1)
    spec = (~pred & ~labels).sum() / max((~labels).sum(), 1)
    return float(sens), float(spec)


# ---------------------------------------------------------------- what the model learns

class CumulativeLogitLoss(nn.Module):
    """
    Learn from the full 0-4 grade instead of only "referable yes/no".

    The network outputs ONE severity score s per eye. Each grade boundary
    k = 0, 1, 2, 3 asks a yes/no question, "is this eye worse than grade k?",
    answered from that same score:

        P(grade > k) = sigmoid(s + cut[k])

    cut[k] are four learnable offsets ("cutpoints"). The loss is ordinary binary
    cross-entropy, averaged over the four questions. This is the classic
    cumulative-logit (proportional-odds) model for ordered labels.

    Why it might help: with the yes/no target, a grade-1 eye (a few
    microaneurysms) is labeled exactly like a perfectly healthy eye, so the
    lesion detector the network is trying to grow gets punished for firing on
    it, and grades 2, 3 and 4 all count the same. Here every grade pulls the
    one score in the same direction (more disease, higher score), and the most
    obvious eyes pull hardest. Whether that beats the yes/no target on real
    EyePACS photos is exactly what exp3 vs exp4 measures.

    The referable probability is P(grade >= 2) = sigmoid(s + cut[1]). Its AUC
    equals the AUC of s itself, so we pick the cutoff on s directly.
    """

    def __init__(self, train_grades, num_grades=NUM_GRADES):
        super().__init__()
        g = np.asarray(train_grades)
        above = np.array([(g > k).mean() for k in range(num_grades - 1)]).clip(1e-3, 1 - 1e-3)
        cut = np.log(above / (1 - above))            # log-odds of each prevalence, decreasing in k
        # Start each cutpoint at the log-odds of its training prevalence, so the
        # untrained network (s ~ 0) already predicts the base rates and the
        # first updates go into learning features, not into finding offsets.
        #
        # The cutpoints must stay in order (cut[0] > cut[1] > ...), or "worse
        # than grade 1" could become more likely than "worse than grade 0".
        # So we learn the first one plus the GAPS between them, and make each
        # gap softplus(raw) + MIN_GAP: softplus is always positive, and the small
        # minimum keeps the gap visible even in float32 arithmetic.
        gaps = np.maximum(cut[:-1] - cut[1:], 2 * self.MIN_GAP)
        self.first = nn.Parameter(torch.tensor(cut[0], dtype=torch.float32))
        self.raw_gaps = nn.Parameter(torch.tensor(np.log(np.expm1(gaps - self.MIN_GAP)),
                                                  dtype=torch.float32))
        self.register_buffer("ks", torch.arange(num_grades - 1))

    MIN_GAP = 1e-3

    def cutpoints(self):
        steps = torch.cumsum(F.softplus(self.raw_gaps) + self.MIN_GAP, dim=0)
        return torch.cat([self.first.view(1), self.first - steps])

    def per_sample(self, score, grade):
        targets = (grade[:, None] > self.ks[None, :]).float()     # e.g. grade 2 -> [1, 1, 0, 0]
        logits = score[:, None] + self.cutpoints()[None, :]
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)

    def forward(self, score, grade):
        return self.per_sample(score, grade).mean()

    def prob_referable(self, score):
        return torch.sigmoid(score + self.cutpoints()[REFERABLE_FROM - 1])


class BinaryLoss(nn.Module):
    """
    The original yes/no target (exp1, exp2): is the grade 2 or worse?

    pos_weight="balanced": most eyes are healthy, so without a correction the
    model could look accurate by always guessing "healthy". The weight makes the
    sick eyes, as a group, count as much as the healthy eyes as a group.
    pos_weight="none": every eye counts the same.
    """

    def __init__(self, train_grades, pos_weight="balanced"):
        super().__init__()
        y = np.asarray(train_grades) >= REFERABLE_FROM
        w = (~y).sum() / max(y.sum(), 1) if pos_weight == "balanced" else 1.0
        self.register_buffer("pos_weight", torch.tensor(float(w)))

    def per_sample(self, score, grade):
        y = (grade >= REFERABLE_FROM).float()
        return F.binary_cross_entropy_with_logits(score, y, pos_weight=self.pos_weight,
                                                  reduction="none")

    def forward(self, score, grade):
        return self.per_sample(score, grade).mean()

    def prob_referable(self, score):
        # With pos_weight="balanced" this is NOT a calibrated probability: the
        # weighting trains the model as if half of all eyes were sick, so the
        # number comes out much higher than the real risk. It still ranks eyes
        # correctly, which is all AUC and the cutoff need.
        return torch.sigmoid(score)


@torch.no_grad()
def no_skill_loss(loss_fn, train_grades, device):
    """
    The loss of a model that knows nothing: the best you can do with one
    constant score for every eye (i.e. only knowing how common each grade is).
    The training loss starts here; how far it drops below this shows how much
    the model has really learned. Found by trying many constants.
    """
    grades_np = np.array(train_grades, dtype=np.int64)      # a writable copy (pandas arrays can be read-only)
    counts = torch.bincount(torch.from_numpy(grades_np), minlength=NUM_GRADES).float()
    consts = torch.linspace(-12, 12, 2401, device=device)
    scores = consts.repeat_interleave(NUM_GRADES)
    grades = torch.arange(NUM_GRADES, device=device).repeat(len(consts))
    per = loss_fn.per_sample(scores, grades).view(len(consts), NUM_GRADES).cpu()
    return float((per @ counts).min() / counts.sum())


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp, memory_format):
    """Score every image once. Returns the mean loss, raw scores and grades."""
    model.eval()                                  # turns off dropout, freezes BatchNorm statistics
    scores, grades, total_loss, n = [], [], 0.0, 0
    for x, g in loader:
        x = x.to(device, non_blocking=True, memory_format=memory_format)
        g = g.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            s = model(x)
        s = s.float()
        total_loss += loss_fn(s, g).item() * len(g)
        n += len(g)
        scores.append(s.cpu())
        grades.append(g.cpu())
    return total_loss / max(n, 1), torch.cat(scores).numpy(), torch.cat(grades).numpy()


def nonfinite_tensors(model):
    """Names of weights or BatchNorm statistics that became NaN/inf (training blew up)."""
    named = list(model.named_parameters()) + list(model.named_buffers())
    return [n for n, t in named if t.is_floating_point() and not torch.isfinite(t).all()]


def save_predictions(path, df, scores, threshold, loss_fn, device):
    """
    One row per eye: who it is, its true grade, the raw score, P(referable), and
    `referred`, the decision this run actually made (score >= threshold).

    `referred` is saved so nobody has to re-apply the threshold to numbers read
    back from text: a score written to CSV and read back can differ in the last
    digit, and the eye sitting exactly at the cutoff could then flip. The score
    is written as a float64; read it with pd.read_csv(path, float_precision="round_trip")
    to get it back exactly.

    Never let a file problem kill a long run: on Windows, a CSV that is open in
    Excel is locked, and writing to it fails.
    """
    with torch.no_grad():
        probs = loss_fn.prob_referable(torch.as_tensor(scores, device=device)).cpu().numpy()
    out = df[["image", "patient", "level"]].copy()
    out["score"] = np.asarray(scores, dtype=np.float64)
    out["prob_referable"] = probs
    out["referred"] = np.asarray(scores) >= threshold
    try:
        out.to_csv(path, index=False)
    except OSError as e:
        print(f"WARNING: could not write {path} ({e}). Is it open in another program?")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="CSV with columns image,level")
    p.add_argument("--images", required=True, help="folder of preprocessed .jpg images")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--overwrite", action="store_true",
                   help="replace results already in --out (otherwise the run refuses to start)")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=None,
                   help="image-loading helper processes (default: CPU count, at most 8)")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--target", choices=["ordinal", "binary"], default="ordinal",
                   help="ordinal: learn the full 0-4 grade; binary: only referable yes/no (exp1/exp2)")
    p.add_argument("--pos-weight", choices=["balanced", "none"], default="balanced",
                   help="binary target only: weight sick eyes up to balance the classes (exp1/exp2)")
    p.add_argument("--rotate-interp", choices=["bilinear", "nearest"], default="bilinear",
                   help="rotation augmentation resampling (exp1/exp2 used nearest)")
    p.add_argument("--rotate-fill", choices=["background", "black"], default="background",
                   help="color of the corners rotation uncovers (exp1/exp2 used black)")
    p.add_argument("--channels-last", type=int, choices=[0, 1], default=1,
                   help="GPU only: memory layout that is often faster for convolutions")
    p.add_argument("--cudnn-benchmark", type=int, choices=[0, 1], default=1,
                   help="GPU only: let cuDNN time its algorithms once and keep the fastest")
    p.add_argument("--limit", type=int, default=0, help="use only about N images, for quick tests")
    p.add_argument("--split-seed", type=int, default=42,
                   help="which patients go to train/val/test (keep 42 to compare runs)")
    p.add_argument("--seed", type=int, default=42,
                   help="weight initialization, shuffling and augmentation")
    args = p.parse_args()

    # Never mix two runs' files in one folder: a leftover test_results.json or
    # best.pt from an earlier run could be mistaken for this run's.
    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in RESULT_FILES if os.path.exists(os.path.join(args.out, f))]
    if existing and not args.overwrite:
        raise SystemExit(f"{args.out} already holds results ({', '.join(existing)}). "
                         "Pick a new --out name, or add --overwrite to replace them.")
    for f in existing:
        os.remove(os.path.join(args.out, f))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"   # mixed precision: faster, less memory, same results on GPU
    memory_format = (torch.channels_last if use_amp and args.channels_last
                     else torch.contiguous_format)
    if use_amp and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True    # input size never changes, so this is safe
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    print(f"Device: {device} | image-loading workers: {workers}")

    # ---------------- Data ----------------
    # Split ALL labeled patients first, then drop eyes whose image is missing.
    # That way the same patients land in validation and test in every run
    # (same --split-seed), even if a few images failed to preprocess.
    df = load_labels(args.labels, referable_from=REFERABLE_FROM)
    parts = dict(zip(["train", "val", "test"], split_by_patient(df, seed=args.split_seed)))
    missing = {}
    for name, part in parts.items():
        exists = part["image"].map(lambda n: os.path.exists(f"{args.images}/{n}.jpg"))
        missing[name] = int((~exists).sum())
        parts[name] = part[exists]
    if sum(missing.values()):
        print(f"Skipping {sum(missing.values())} labels with no image file: {missing}")
    if args.limit:
        frac = min(1.0, args.limit / len(df))
        parts = {k: v.sample(frac=frac, random_state=args.split_seed) for k, v in parts.items()}
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise SystemExit("A split is empty: use more images (raise --limit).")
    for name, part in parts.items():
        part.to_csv(f"{args.out}/{name}_split.csv", index=False)
        print(f"{name:5s}: {len(part):6d} images, {part['label'].mean():.1%} referable")

    mean, std = compute_mean_std(train_df, args.images)
    background = estimate_background(train_df, args.images)
    fill = background if args.rotate_fill == "background" else (0, 0, 0)
    enhanced = all(abs(c - 128) <= 10 for c in background)   # --enhance leaves a gray background
    print(f"Background color outside the eye: {background}"
          f" ({'enhanced' if enhanced else 'plain'} preprocessing)")
    train_tf, eval_tf = make_transforms(args.size, mean, std, fill, args.rotate_interp)

    def loader(part, tf, n_workers, shuffle=False, keep_workers=False):
        # persistent_workers keeps the helper processes alive between epochs.
        # Starting them is slow on Windows (each one loads PyTorch again), so
        # without it every epoch would begin with a long pause.
        return DataLoader(FundusDataset(part, args.images, tf), batch_size=args.batch,
                          shuffle=shuffle, drop_last=shuffle and len(part) > args.batch,
                          num_workers=n_workers, pin_memory=use_amp,
                          persistent_workers=keep_workers and n_workers > 0)

    eval_workers = max(1, workers // 2) if workers else 0
    train_loader = loader(train_df, train_tf, workers, shuffle=True, keep_workers=True)
    val_loader = loader(val_df, eval_tf, eval_workers, keep_workers=True)
    # (the test loader is only created at the very end, for the single test pass)

    # ---------------- Model, loss, optimizer ----------------
    model = RetinaResNet().to(device, memory_format=memory_format)
    train_grades = train_df["level"].to_numpy()
    if args.target == "ordinal":
        loss_fn = CumulativeLogitLoss(train_grades).to(device)
    else:
        loss_fn = BinaryLoss(train_grades, args.pos_weight).to(device)
    no_skill = no_skill_loss(loss_fn, train_grades, device)
    print(f"Target: {args.target} | loss of a model that knows nothing: {no_skill:.4f}")

    commit, dirty = code_version()
    config = dict(vars(args), workers_used=workers, code_version=commit, code_dirty=dirty,
                  torch=torch.__version__, device=str(device), mean=mean, std=std,
                  background=background, enhanced_inferred=enhanced, no_skill_loss=no_skill,
                  n_images={k: len(v) for k, v in parts.items()}, missing_images=missing,
                  grade_counts={k: v["level"].value_counts().sort_index().to_dict()
                                for k, v in parts.items()},
                  sanity_run=bool(args.limit))
    write_json(f"{args.out}/config.json", config)

    # Weight decay gently pulls weights toward zero, which reduces overfitting.
    # It's applied to the conv/linear weights, not to BatchNorm, bias terms or
    # the loss's cutpoints.
    decay = [q for q in model.parameters() if q.ndim >= 2]
    no_decay = [q for q in model.parameters() if q.ndim < 2] + list(loss_fn.parameters())
    params = list(model.parameters()) + list(loss_fn.parameters())   # for gradient clipping
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
    best_auc, history, blew_up = -1.0, [], False
    best_val_scores, best_thr = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen, skipped, waited = 0.0, 0, 0, 0.0
        train_scores, train_grades_seen = [], []
        t_ready = time.time()
        for x, g in train_loader:
            waited += time.time() - t_ready             # time spent waiting for the next batch
            x = x.to(device, non_blocking=True, memory_format=memory_format)
            g = g.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                s = model(x)                               # 1. forward: make predictions
            loss = loss_fn(s.float(), g)                   # 2. measure how wrong they are
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()                  # 3. backward: compute gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)          # stop rare huge updates
            scaler.step(optimizer)                         # 4. nudge every weight downhill
            scaler.update()
            scheduler.step()
            # .item() waits for the GPU to finish this step. That also keeps the
            # "waiting for data" timer honest; if you ever remove it, put
            # torch.cuda.synchronize() here instead.
            loss_value = loss.item()
            if math.isfinite(loss_value):
                running += loss_value * len(g)
                seen += len(g)
            else:                                          # mixed precision skips such steps
                skipped += 1
            train_scores.append(s.detach().float().cpu())
            train_grades_seen.append(g.cpu())
            t_ready = time.time()
        train_seconds = time.time() - t0

        val_loss, val_scores, val_grades = evaluate(model, val_loader, loss_fn, device,
                                                    use_amp, memory_format)
        if not np.isfinite(val_scores).all():
            bad = nonfinite_tensors(model)
            print(f"epoch {epoch}: validation scores are NaN/inf, training has blown up. "
                  f"Broken tensors: {bad[:5] or 'none (only some activations overflowed)'}. "
                  "Try a lower --lr. Stopping; if a checkpoint was saved, it is still tested.")
            blew_up = True
            break
        val_labels = val_grades >= REFERABLE_FROM
        val_auc = safe_auc(val_labels, val_scores)

        # Train AUC is measured on the fly: augmented images, dropout on, weights
        # still changing during the epoch. So it normally sits BELOW val AUC.
        # Overfitting shows up as train AUC clearly above val AUC late in training.
        ts, tg = torch.cat(train_scores).numpy(), torch.cat(train_grades_seen).numpy()
        finite = np.isfinite(ts)
        train_loss = running / max(seen, 1)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   train_auc=safe_auc(tg[finite] >= REFERABLE_FROM, ts[finite]),
                   val_auc=val_auc, minutes=(time.time() - t0) / 60,
                   lr=scheduler.get_last_lr()[0], skipped_steps=skipped,
                   train_img_per_s=len(tg) / train_seconds, data_wait=waited / train_seconds,
                   val_mean_score_by_grade={int(k): float(val_scores[val_grades == k].mean())
                                            for k in np.unique(val_grades)})
        msg = (f"epoch {epoch:3d} | train loss {train_loss:.4f} ({train_loss / no_skill:.0%} "
               f"of no-skill) | val loss {val_loss:.4f} | val AUC {val_auc:.4f}")
        if not math.isnan(val_auc):
            thr = pick_threshold(val_labels, val_scores, args.target_sensitivity)
            sens, spec = sens_spec(val_labels, val_scores, thr)
            row["val_specificity"] = spec
            msg += f" | specificity at {sens:.0%} sensitivity: {spec:.1%}"
        print(msg + f" | {row['minutes']:.1f} min")
        # Mean raw score per grade: once the model is learning lesions, these
        # climb from grade 0 to grade 4 (only their order matters, not the values).
        grade_means = " ".join(f"g{k} {v:+.2f}" for k, v in row["val_mean_score_by_grade"].items())
        print(f"          train AUC {row['train_auc']:.3f} | lr {row['lr']:.1e} | "
              f"{row['train_img_per_s']:.0f} img/s, waiting for data {row['data_wait']:.0%} | "
              f"mean val score: {grade_means}")
        if epoch == 1:
            print(f"          (about {row['minutes'] * args.epochs:.0f} min for all "
                  f"{args.epochs} epochs at this speed)")
        history.append(row)

        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "loss": loss_fn.state_dict(),
                        **to_py({"target": args.target, "pos_weight": args.pos_weight,
                                 "mean": mean, "std": std, "size": args.size,
                                 "background": background, "threshold": thr,
                                 "epoch": epoch, "val_auc": val_auc})},
                       f"{args.out}/best.pt")
            # Keep the exact validation scores this epoch's cutoff was picked
            # on; they are written to val_predictions.csv after training.
            best_val_scores, best_thr = val_scores.copy(), thr
            print("    new best, checkpoint saved")
        write_json(f"{args.out}/history.json", history)

    # ---------------- Final test ----------------
    # The test set is touched exactly once, at the very end, using the model and
    # cutoff chosen on validation data. Peeking at it earlier would make the
    # final number look better than reality.
    #
    # Ask whether THIS run saved a checkpoint (best_auc only moves above -1
    # when it does). Checking whether best.pt exists is not enough: a file
    # left over from an earlier run in the same --out folder would be loaded
    # and tested instead, and that model may have trained on these test eyes.
    if best_auc < 0:
        if blew_up:
            print("Training blew up before any checkpoint was saved. Try a lower --lr.")
        else:
            print("No checkpoint was saved (validation set needs both classes). Use more data.")
        return
    save_predictions(f"{args.out}/val_predictions.csv", val_df, best_val_scores, best_thr,
                     loss_fn, device)

    # Stop the training and validation helper processes before testing, so
    # their memory is free; the test loader starts its own (fewer) helpers.
    del train_loader, val_loader
    test_loader = loader(test_df, eval_tf, eval_workers)
    ckpt = torch.load(f"{args.out}/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    loss_fn.load_state_dict(ckpt["loss"])
    _, test_scores, test_grades = evaluate(model, test_loader, loss_fn, device,
                                           use_amp, memory_format)
    test_labels = test_grades >= REFERABLE_FROM
    test_auc = safe_auc(test_labels, test_scores)
    sens, spec = sens_spec(test_labels, test_scores, ckpt["threshold"])
    # Save the test results right away, before anything else can go wrong:
    # re-running to get them back would mean looking at the test set twice.
    # Any further analysis (confidence intervals, per-grade, per-patient) can be
    # done later from test_predictions.csv, without the model or the images.
    write_json(f"{args.out}/test_results.json",
               {"auc": test_auc, "sensitivity": sens, "specificity": spec,
                "threshold": ckpt["threshold"], "best_epoch": ckpt["epoch"],
                "best_val_auc": ckpt["val_auc"], "target": args.target,
                "sanity_run": bool(args.limit)})
    save_predictions(f"{args.out}/test_predictions.csv", test_df, test_scores,
                     ckpt["threshold"], loss_fn, device)
    if args.limit:
        print("\nSANITY RUN (--limit): a quick check that everything works, NOT a result.")
    print(f"\nTEST (best epoch {ckpt['epoch']}): AUC {test_auc:.4f} | "
          f"sensitivity {sens:.1%} | specificity {spec:.1%}")


if __name__ == "__main__":
    main()
