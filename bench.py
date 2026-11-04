"""
Speed test: find out what limits training speed on this machine.

It times three things separately, each for a few dozen batches:
  1. GPU only:      training steps on random images already on the GPU
                    (how fast the GPU could go if images were free)
  2. Loading only:  reading + augmenting real images with N helper processes,
                    no GPU at all (how fast the CPU can feed the GPU)
  3. Together:      real images through real training steps, like train.py

Whichever of 1 and 2 is slower is the bottleneck, and 3 should be close to it.
If 3 is much slower than both, something else on the machine is in the way
(memory swapping, other programs, heat).

Stop any running training first (it would compete for the GPU), then:
  python bench.py --images %USERPROFILE%\\retina-data\\processed_384_enhanced
Takes about 3-4 minutes on a laptop GPU.
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FundusDataset, make_transforms
from model import build_model
from train import CumulativeLogitLoss


def timed_steps(n_steps, step_fn, warmup=5, sync=None):
    """Run step_fn warmup + n_steps times; return seconds for the last n_steps."""
    for _ in range(warmup):
        step_fn()
    if sync:
        sync()
    t0 = time.time()
    for _ in range(n_steps):
        step_fn()
    if sync:
        sync()
    return time.time() - t0


def make_train_step(device, use_amp, channels_last, batch, size):
    """One training step exactly like train.py's: AMP, GradScaler, clipping, AdamW."""
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model = build_model().to(device, memory_format=fmt).train()
    loss_fn = CumulativeLogitLoss([0] * 70 + [1] * 8 + [2] * 15 + [3] * 4 + [4] * 3).to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def step(x, g):
        x = x.to(device, non_blocking=True, memory_format=fmt)
        g = g.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            s = model(x)
        loss = loss_fn(s.float(), g)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        return loss.item()                       # waits for the GPU, like train.py
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder of preprocessed .jpg images")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=30, help="batches to time per test")
    p.add_argument("--workers", type=int, nargs="+", default=[4, 8],
                   help="helper counts to try for loading")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    sync = torch.cuda.synchronize if use_amp else None
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}")
    else:
        print("No GPU found: running on the CPU (numbers will be tiny).")
    print(f"CPU threads: {os.cpu_count()} | image size {args.size} | batch {args.batch}\n")

    names = sorted(f[:-4] for f in os.listdir(args.images) if f.endswith(".jpg"))
    df = pd.DataFrame({"image": names[: args.batch * (args.steps + 10) * 2], "level": 0})
    train_tf, _ = make_transforms(args.size, [0.5] * 3, [0.25] * 3, (128, 128, 128))
    n = args.steps * args.batch

    # 1. GPU only: the same random batch over and over, already in place.
    results = {}
    for cl in ([1, 0] if use_amp else [0]):
        step = make_train_step(device, use_amp, cl, args.batch, args.size)
        x = torch.randn(args.batch, 3, args.size, args.size).pin_memory() if use_amp else \
            torch.randn(args.batch, 3, args.size, args.size)
        g = torch.randint(0, 5, (args.batch,))
        secs = timed_steps(args.steps, lambda: step(x, g), sync=sync)
        results[f"gpu_cl{cl}"] = n / secs
        print(f"1. GPU only (channels_last={cl}):        {n / secs:7.0f} img/s")

    # 2. Loading only: real images, real augmentation, no GPU work.
    for w in args.workers:
        loader = DataLoader(FundusDataset(df, args.images, train_tf), batch_size=args.batch,
                            shuffle=True, num_workers=w, pin_memory=use_amp,
                            persistent_workers=w > 0, drop_last=True)
        it = iter(loader)
        for _ in range(3):                       # let the helpers start up
            next(it)
        t0 = time.time()
        for _ in range(args.steps):
            next(it)
        secs = time.time() - t0
        results[f"load_w{w}"] = n / secs
        print(f"2. Loading only ({w} helpers):             {n / secs:7.0f} img/s")
        del it, loader

    # 3. Together, like train.py (with the largest helper count).
    w = max(args.workers)
    loader = DataLoader(FundusDataset(df, args.images, train_tf), batch_size=args.batch,
                        shuffle=True, num_workers=w, pin_memory=use_amp,
                        persistent_workers=w > 0, drop_last=True)
    step = make_train_step(device, use_amp, 1 if use_amp else 0, args.batch, args.size)
    it = iter(loader)
    for _ in range(3):
        step(*next(it))
    waited, t0 = 0.0, time.time()
    for _ in range(args.steps):
        t = time.time()
        x, g = next(it)
        waited += time.time() - t
        step(x, g)
    secs = time.time() - t0
    print(f"3. Together ({w} helpers):                 {n / secs:7.0f} img/s"
          f"  (waiting for data {waited / secs:.0%} of the time)")

    gpu = max(v for k, v in results.items() if k.startswith("gpu"))
    load = max(v for k, v in results.items() if k.startswith("load"))
    print("\nReading it:")
    print(f"  the GPU could do ~{gpu:.0f} img/s; the CPU can load ~{load:.0f} img/s.")
    print(f"  Expected together: about {min(gpu, load):.0f} img/s; measured {n / secs:.0f}.")
    print("  (A full epoch is ~24,600 training images.)")


if __name__ == "__main__":
    main()
