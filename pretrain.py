"""
Stage 2: self-supervised pretraining, written from scratch (SimMIM-style masked
image modeling) for the SAME network as Stage 1.

The game: hide ~60% of each photo in 32x32 squares, and make the network redraw
what was hidden. To fill in a hidden vessel it has to know how vessels run; to
guess a hidden lesion it has to understand the retina around it. No labels are
needed, so it can practice on every photo we have.

    photo -> hide random squares -> ResNet (stem + stages, exactly as in Stage 1)
          -> 16x16 feature map (at 512 px) -> tiny decoder: one 1x1 conv that
             predicts 32x32x3 pixel values per feature cell, then PixelShuffle
             lays them out as a full 512x512 image
          -> loss = average |prediction - original| on the HIDDEN pixels only

Afterwards the decoder is thrown away. encoder.pt (the stem and stages weights)
is the starting point for disease training:
    python train.py ... --init runs\\pre1\\encoder.pt

Which photos: any folders of preprocessed images (e.g. the unused EyePACS test
folder plus the training images). The labeled VALIDATION and TEST patients are
always left out (--exclude-labels + --split-seed rebuild exactly the same split
as train.py), so Stage 2 can be compared fairly with Stage 1. Labels are never read
for anything else.

Usage:
  python pretrain.py --images DIR [DIR ...] --exclude-labels trainLabels.csv --out runs\\pre1
  python pretrain.py ... --resume          (continue after an interruption)
Every few epochs, <out>\\recon_epochNN.png shows: original | what the network saw | its reconstruction.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import load_labels, make_transforms, split_by_patient
from model import build_model


class UnlabeledImages(Dataset):
    """Just images: the label is never needed here."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class SimMIM(nn.Module):
    """
    Encoder = the Stage 1 network's stem and stages (its pooling and head are not
    used here). A learnable "mask color" replaces the hidden squares in the input;
    the decoder is one 1x1 convolution + PixelShuffle.
    """

    def __init__(self, depth=18, pool="avg", patch=32):
        super().__init__()
        self.net = build_model(depth=depth, pool=pool)   # same builder as train.py
        self.patch = patch
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))   # the learned "hidden" color
        width = self.net.head.in_features                # 512 channels after the last stage
        stride = 32                                      # the network shrinks images 32x
        self.decoder = nn.Sequential(nn.Conv2d(width, stride * stride * 3, kernel_size=1),
                                     nn.PixelShuffle(stride))

    def random_mask(self, x, ratio):
        """1 where a 32x32 square is hidden, 0 where visible; shape (B, 1, H, W)."""
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        n_hide = int(round(ratio * gh * gw))
        order = torch.rand(b, gh * gw, device=x.device).argsort(dim=1)
        grid = torch.zeros(b, gh * gw, device=x.device)
        grid.scatter_(1, order[:, :n_hide], 1.0)          # the first n_hide squares in a random order
        grid = grid.view(b, 1, gh, gw)
        return grid.repeat_interleave(self.patch, 2).repeat_interleave(self.patch, 3)

    def forward(self, x, ratio):
        mask = self.random_mask(x, ratio)
        seen = x * (1 - mask) + self.mask_token * mask    # hidden squares get the mask color
        features = self.net.stages(self.net.stem(seen))
        recon = self.decoder(features)
        # Only the hidden pixels count: reconstructing what it can already see teaches nothing.
        err = (recon.float() - x.float()).abs() * mask
        loss = err.sum() / (mask.sum() * x.shape[1]).clamp(min=1)
        return loss, recon, seen, mask

    def encoder_state(self):
        """The weights train.py --init loads: stem and stages only."""
        return {k: v for k, v in self.net.state_dict().items() if k.startswith(("stem.", "stages."))}


def list_images(folders, exclude_labels, split_seed):
    """All .jpg images in the folders, minus every image of a validation or test patient."""
    excluded = set()
    if exclude_labels:
        _, val_df, test_df = split_by_patient(load_labels(exclude_labels), seed=split_seed)
        excluded = set(val_df["patient"]) | set(test_df["patient"])
    paths, dropped = [], 0
    for folder in folders:
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".jpg"):
                continue
            patient = name[:-4].split("_")[0]
            if patient in excluded:
                dropped += 1
                continue
            paths.append(os.path.join(folder, name))
    return paths, dropped, len(excluded)


def mean_std(paths, n=500, seed=0):
    """Average color and spread of a sample of the pretraining images."""
    sample = random.Random(seed).sample(paths, min(n, len(paths)))
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    for path in sample:
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        total += a.sum(axis=(0, 1))
        total_sq += (a ** 2).sum(axis=(0, 1))
        count += a.shape[0] * a.shape[1]
    mean = total / count
    return mean.tolist(), np.sqrt(total_sq / count - mean ** 2).tolist()


def save_examples(path, model, batch, ratio, mean, std):
    """
    One picture with three rows of the same few photos:
      top: the original | middle: what the network saw | bottom: its reconstruction.
    """
    model.eval()
    with torch.no_grad():
        _, recon, seen, mask = model(batch, ratio)
        recon = recon * mask + batch * (1 - mask)          # show predictions only where hidden
    m = torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    rows = []
    for tensor in (batch, seen, recon):
        img = (tensor.float() * s + m).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        rows.append(np.concatenate(list(img), axis=1))     # the photos side by side
    picture = np.concatenate(rows, axis=0)                 # the three rows stacked
    Image.fromarray((picture * 255).astype(np.uint8)).save(path)
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True, help="folders of preprocessed .jpg images")
    p.add_argument("--exclude-labels", required=True,
                   help="the labeled CSV; its validation and test patients are left out")
    p.add_argument("--split-seed", type=int, default=42, help="must match train.py's (default 42)")
    p.add_argument("--out", default="runs/pre1")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.6, help="share of 32x32 squares hidden")
    p.add_argument("--depth", type=int, choices=[18, 34], default=18)
    p.add_argument("--pool", choices=["avg", "gem"], default="gem",
                   help="must match the later train.py --pool (not used during pretraining itself)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="use only N images, for quick tests")
    p.add_argument("--example-every", type=int, default=5, help="save reconstruction pictures every N epochs")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    last_path = os.path.join(args.out, "last.pt")
    if os.path.exists(os.path.join(args.out, "history.json")) and not args.resume:
        raise SystemExit(f"{args.out} already holds a pretraining run. Use --resume to continue it, "
                         "or pick a new --out.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)

    paths, dropped, n_excluded = list_images(args.images, args.exclude_labels, args.split_seed)
    if args.limit:
        paths = random.Random(args.seed).sample(paths, min(args.limit, len(paths)))
    print(f"Device: {device} | {len(paths)} images for pretraining | left out {dropped} images of the "
          f"{n_excluded} validation/test patients")
    if not paths:
        raise SystemExit("No images found.")
    mean, std = mean_std(paths)
    train_tf, _ = make_transforms(args.size, mean, std, (128, 128, 128))
    loader = DataLoader(UnlabeledImages(paths, train_tf), batch_size=args.batch, shuffle=True,
                        drop_last=len(paths) > args.batch, num_workers=workers, pin_memory=use_amp,
                        persistent_workers=workers > 0)

    model = SimMIM(args.depth, args.pool).to(device)
    decay = [q for n, q in model.named_parameters() if q.ndim >= 2]
    no_decay = [q for n, q in model.named_parameters() if q.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total_steps = args.epochs * max(len(loader), 1)
    warmup = max(1, min(len(loader), total_steps // 10))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, history = 1, []
    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume: {last_path} not found.")
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, history = state["epoch"] + 1, state["history"]
        print(f"Resuming after epoch {state['epoch']}")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), workers_used=workers, mean=mean, std=std, n_images=len(paths),
                       n_left_out=dropped), f, indent=2)
    examples = next(iter(DataLoader(UnlabeledImages(paths[:4], train_tf), batch_size=4))).to(device)

    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, _, _, _ = model(x, args.mask_ratio)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            value = loss.item()
            if math.isfinite(value):
                total += value * len(x)
                seen += len(x)
        row = dict(epoch=epoch, loss=total / max(seen, 1), lr=scheduler.get_last_lr()[0],
                   minutes=(time.time() - t0) / 60, img_per_s=seen / (time.time() - t0))
        history.append(row)
        print(f"epoch {epoch:3d} | reconstruction loss {row['loss']:.4f} | lr {row['lr']:.1e} | "
              f"{row['img_per_s']:.0f} img/s | {row['minutes']:.1f} min")
        if epoch == start:
            print(f"          (about {row['minutes'] * (args.epochs - start + 1):.0f} min for the remaining epochs)")
        # Save every epoch: the encoder for train.py --init, and everything needed to resume.
        torch.save({"encoder": model.encoder_state(),
                    "arch": {"depth": args.depth, "pool": args.pool}, "epoch": epoch},
                   os.path.join(args.out, "encoder.pt"))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history}, last_path)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if epoch % args.example_every == 0 or epoch == args.epochs:
            save_examples(os.path.join(args.out, f"recon_epoch{epoch:02d}.png"), model, examples,
                          args.mask_ratio, mean, std)
    print(f"Done. Encoder for train.py --init: {os.path.join(args.out, 'encoder.pt')}")


if __name__ == "__main__":
    main()
