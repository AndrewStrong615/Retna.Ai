"""
Make a tiny FAKE fundus dataset for smoke-testing the pipeline on a CPU.

These pictures are cartoons, not retinas. They exist for one reason: to check
that every script runs end to end (preprocess -> train -> test) in a few
minutes, before spending hours of GPU time on the real data.

The disease is drawn as obvious spots (dark red dots for microaneurysms and
hemorrhages, yellow blobs for exudates), with more spots at higher grades.
That makes the task easy on purpose. A working pipeline should push the
validation AUC above 0.9 within about 20 epochs. If labels ever got shuffled
relative to images (a classic silent bug), the AUC would stay near 0.5 instead.

The output imitates EyePACS so the real scripts can't tell the difference:
  <out>/raw/{patient}_{left|right}.jpeg   wide photos, black border, round eye
  <out>/trainLabels.csv                   columns image,level (grade 0-4)

One file is deliberately unreadable, to exercise the error handling in
preprocess.py ("1 images failed") and train.py ("Skipping 1 labels...").

Usage (about 30 seconds on a CPU; checks that everything RUNS):
  python make_synthetic_data.py --out synthetic
  python preprocess.py --src synthetic/raw --dst synthetic/processed_128 --size 128
  python train.py --labels synthetic/trainLabels.csv --images synthetic/processed_128 \\
                  --size 128 --epochs 3 --batch 16 --workers 2 --out runs/smoke

After 3 epochs the model is still close to random, so ignore the AUC there.
To check that it LEARNS, run the same train.py command with --epochs 20
(about 2 minutes on 4 CPU cores): validation AUC should climb past 0.9.
train.py refuses to reuse a folder that holds results: add --overwrite (or
pick a new --out) to run it again. For a quick automated check of everything,
use `python selftest.py` instead.
"""

import argparse
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Share of patients at each grade 0..4. Real EyePACS is about 73/7/15/2.5/2
# percent (about 20% referable). We use a less lopsided mix so that even a
# tiny validation set contains both classes and AUC can be computed.
GRADE_PROBS = [0.45, 0.15, 0.25, 0.08, 0.07]


def draw_eye(rng, level):
    """Draw one fake fundus photo with lesions matching the grade."""
    # Real photos come from different cameras: vary size, framing and exposure.
    width, height = rng.randint(480, 640), rng.randint(360, 480)
    img = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    # The eye is a circle. Often it is taller than the photo and gets cut off
    # at the top and bottom, as in many EyePACS images.
    r = int(min(width * 0.45, height * rng.uniform(0.45, 0.6)))
    cx, cy = width // 2 + rng.randint(-15, 15), height // 2
    exposure = rng.uniform(0.7, 1.2)

    def shade(rgb, k=1.0):
        return tuple(int(min(255, c * k * exposure)) for c in rgb)

    background = (rng.randint(160, 200), rng.randint(60, 90), rng.randint(20, 40))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade(background))

    # Optic disc: a bright spot to one side, where the vessels start.
    side = rng.choice([-1, 1])
    disc = (cx + side * int(r * 0.45), cy + rng.randint(-r // 8, r // 8))
    disc_r = int(r * 0.12)
    d.ellipse([disc[0] - disc_r, disc[1] - disc_r, disc[0] + disc_r, disc[1] + disc_r],
              fill=shade((240, 200, 140)))

    # Blood vessels: dark red random walks leaving the optic disc.
    for _ in range(rng.randint(5, 8)):
        angle = rng.uniform(0, 2 * math.pi)
        x, y = disc
        points = [(x, y)]
        for _ in range(25):
            angle += rng.uniform(-0.3, 0.3)
            x += math.cos(angle) * r * 0.06
            y += math.sin(angle) * r * 0.06
            points.append((x, y))
        d.line(points, fill=shade(background, 0.55), width=rng.randint(3, 5), joint="curve")

    def random_spot_center():
        # Uniform inside 85% of the eye circle (sqrt keeps the density even).
        a, rr = rng.uniform(0, 2 * math.pi), r * 0.85 * math.sqrt(rng.random())
        return cx + rr * math.cos(a), cy + rr * math.sin(a)

    # Lesions. Counts rise with grade; exudates first appear at grade 2,
    # which is also where "referable" begins, so the label is learnable.
    n_red = [0, rng.randint(1, 3), rng.randint(5, 9), rng.randint(9, 15), rng.randint(12, 18)][level]
    n_yellow = [0, 0, rng.randint(1, 3), rng.randint(3, 6), rng.randint(4, 8)][level]
    for _ in range(n_red):
        x, y = random_spot_center()
        s = rng.uniform(r * 0.012, r * 0.035)
        d.ellipse([x - s, y - s, x + s, y + s], fill=shade(background, 0.35))
    for _ in range(n_yellow):
        x, y = random_spot_center()
        s = rng.uniform(r * 0.02, r * 0.05)
        d.ellipse([x - s, y - s, x + s, y + s], fill=shade((235, 215, 90)))

    # Soften the drawing, then add sensor noise inside the eye only, so the
    # border stays truly black (preprocess.py finds the eye by brightness).
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    a = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, a.shape)
    a[inside] += noise[inside]
    a[~inside] = 0
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="synthetic", help="output folder")
    p.add_argument("--patients", type=int, default=100, help="each patient has 2 eyes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # EyePACS patient numbers are unique but not consecutive (10, 13, 15, ...).
    patients = sorted(rng.sample(range(1, 50_000), args.patients))
    rows = []
    for pid in patients:
        # Both eyes of one person tend to have similar disease. That is exactly
        # why the train/val/test split must be done by patient.
        base = rng.choices(range(5), weights=GRADE_PROBS)[0]
        for eye in ("left", "right"):
            level = base if rng.random() < 0.7 else min(4, max(0, base + rng.choice([-1, 1])))
            name = f"{pid}_{eye}"
            draw_eye(rng, level).save(os.path.join(raw_dir, name + ".jpeg"), quality=90)
            rows.append((name, level))

    # Overwrite one image with junk bytes: its label stays in the CSV, but
    # preprocessing will fail on it and training must skip it gracefully.
    broken = rows[0][0]
    with open(os.path.join(raw_dir, broken + ".jpeg"), "wb") as f:
        f.write(b"this is not a jpeg")

    with open(os.path.join(args.out, "trainLabels.csv"), "w") as f:
        f.write("image,level\n")
        f.writelines(f"{name},{level}\n" for name, level in rows)

    n_ref = sum(level >= 2 for _, level in rows)
    print(f"Wrote {len(rows)} images from {len(patients)} patients to {raw_dir}")
    print(f"  {n_ref} referable (grade >= 2), {len(rows) - n_ref} not")
    print(f"  {broken}.jpeg is deliberately corrupt, to test error handling")
    print(f"Labels: {os.path.join(args.out, 'trainLabels.csv')}")


if __name__ == "__main__":
    main()
