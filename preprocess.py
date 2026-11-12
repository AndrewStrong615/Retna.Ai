"""
Step 1: Preprocess raw fundus photos ONCE, before training.

Raw EyePACS images are huge (often 3000+ pixels wide) and come from many
different cameras. Decoding them every epoch would make training painfully
slow, so we shrink them once and save the results.

For each image we:
  1. Crop away the black border around the circular eye.
  2. Pad it into a square (so resizing doesn't squash the eye into an oval).
  3. Resize to a fixed size (384x384 by default).
  4. (Optional) Apply "local average color subtraction", a trick popularized
     in the 2015 Kaggle diabetic retinopathy competition. It removes lighting
     differences between cameras and makes small lesions stand out.

Usage:
  python preprocess.py --src path/to/raw_images --dst path/to/processed --size 384
  python preprocess.py --src ... --dst ... --size 384 --enhance
  python preprocess.py --src ... --dst ... --size 512 --enhance --tight-mask

--tight-mask (new, off by default so earlier folders stay reproducible): instead
of assuming the eye is a perfect circle, find each photo's real eye outline,
trim a thin band at its edge, and make everything outside it background. With
--enhance it also blurs only the eye itself, so the dark area outside can't leak
in and create the bright rim that heat maps showed the model looking at.

Each output folder also gets preprocess.json with the settings used.
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter


def crop_black_border(img, threshold=10):
    """Remove the dark frame around the eye by finding where pixels are bright."""
    gray = np.asarray(img.convert("L"))
    mask = gray > threshold                      # True wherever there's eye, not background
    if not mask.any():                           # fully black image (bad photo): leave as-is
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def pad_to_square(img):
    """Center the image on a black square canvas."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def eye_mask(img, margin, threshold=10):
    """
    True inside the eye, False outside, for an already cropped, padded and
    resized photo.

    1. "Bright enough" pixels (same test as crop_black_border) mark the eye.
    2. Each row is filled between its first and last bright pixel, so dark
       spots INSIDE the eye (hemorrhages!) are never mistaken for outside.
    3. The mask is shrunk by `margin` pixels all round, to drop the ragged,
       darker rim at the very edge of the photo. Shrinking uses a box blur:
       after blurring, only pixels whose whole neighborhood was inside stay at 255.
    """
    bright = np.asarray(img.convert("L")) > threshold
    has_eye = bright.any(axis=1)
    first = bright.argmax(axis=1)
    last = bright.shape[1] - 1 - bright[:, ::-1].argmax(axis=1)
    cols = np.arange(bright.shape[1])
    mask = has_eye[:, None] & (cols[None, :] >= first[:, None]) & (cols[None, :] <= last[:, None])
    if margin > 0:
        # Pad with "outside" first, so the edge of the image also counts as outside
        # (otherwise the eye would not shrink where it touches the image border).
        padded = np.pad(mask, margin + 1, constant_values=False)
        blurred = Image.fromarray(padded.astype(np.uint8) * 255).filter(ImageFilter.BoxBlur(margin))
        mask = (np.asarray(blurred) >= 254)[margin + 1:-(margin + 1), margin + 1:-(margin + 1)]
    return mask


def enhance(img, mask=None):
    """
    Local average color subtraction:  out = 4 * image - 4 * blurred_image + 128

    The blurred image captures slow changes in brightness and color (camera
    lighting, pigmentation). Subtracting it leaves the fine details, like
    tiny red dots (microaneurysms) and bright spots (exudates), on a neutral
    gray background. We then gray out everything outside the eye circle,
    where the blur creates a bright ring artifact.

    With a `mask` (from --tight-mask), the blur only averages pixels INSIDE the
    eye: blur(image * mask) / blur(mask). Near the edge, the black outside no
    longer drags the average down, so no bright ring forms. Everything outside
    the mask becomes gray.
    """
    size = img.size[0]
    radius = size / 30
    a = np.asarray(img, dtype=np.float32)
    if mask is None:
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        yy, xx = np.mgrid[:size, :size]
        center = (size - 1) / 2
        outside = (yy - center) ** 2 + (xx - center) ** 2 > (0.9 * size / 2) ** 2
    else:
        inside = Image.fromarray((a * mask[..., None]).astype(np.uint8))
        weight = Image.fromarray(mask.astype(np.uint8) * 255)
        num = np.asarray(inside.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        den = np.asarray(weight.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        b = num / np.maximum(den / 255.0, 1e-3)[..., None]
        outside = ~mask
    out = np.clip(4 * a - 4 * b + 128, 0, 255)
    out[outside] = 128
    return Image.fromarray(out.astype(np.uint8))


def process_one(job):
    src_path, dst_path, size, do_enhance, tight_mask, mask_margin = job
    try:
        img = Image.open(src_path).convert("RGB")
        img = pad_to_square(crop_black_border(img))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        mask = eye_mask(img, margin=round(size * mask_margin)) if tight_mask else None
        if do_enhance:
            img = enhance(img, mask)
        elif mask is not None:                   # plain images: just blacken outside the eye
            a = np.asarray(img).copy()
            a[~mask] = 0
            img = Image.fromarray(a)
        img.save(dst_path, quality=95)
        return None
    except Exception as e:                       # some dataset images are corrupt
        return f"{src_path}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, nargs="+",
                   help="folder(s) of raw images (several = one output folder, e.g. DDR's train valid test)")
    p.add_argument("--dst", required=True, help="output folder")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--enhance", action="store_true", help="apply local average color subtraction")
    p.add_argument("--tight-mask", action="store_true",
                   help="use each photo's real eye outline instead of a fixed circle")
    p.add_argument("--mask-margin", type=float, default=0.02,
                   help="with --tight-mask: trim this fraction of the image size at the eye's edge")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    exts = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
    jobs, seen = [], {}
    for src in args.src:
        for name in sorted(os.listdir(src)):
            if name.lower().endswith(exts):
                stem = os.path.splitext(name)[0]
                # Two photos with the same name (from two source folders) would overwrite
                # each other in the output folder: refuse instead of silently losing one.
                if stem in seen:
                    raise SystemExit(f"{name} is in both {seen[stem]} and {src}: names must be unique.")
                seen[stem] = src
                jobs.append((os.path.join(src, name),
                             os.path.join(args.dst, stem + ".jpg"),
                             args.size, args.enhance, args.tight_mask, args.mask_margin))

    print(f"Processing {len(jobs)} images with {args.workers} workers...")
    errors = []
    with Pool(args.workers) as pool:
        for i, err in enumerate(pool.imap_unordered(process_one, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)} done")
    print(f"Finished. {len(errors)} images failed.")
    for e in errors[:20]:
        print("  ", e)
    # Record how this folder was made; train.py copies it into each run's config.json.
    with open(os.path.join(args.dst, "preprocess.json"), "w") as f:
        json.dump({"size": args.size, "enhance": args.enhance, "tight_mask": args.tight_mask,
                   "mask_margin": args.mask_margin if args.tight_mask else None,
                   "n_images": len(jobs), "n_failed": len(errors), "failed": errors[:100]},
                  f, indent=2)


if __name__ == "__main__":
    main()
