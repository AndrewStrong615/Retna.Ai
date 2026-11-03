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
"""

import argparse
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


def enhance(img):
    """
    Local average color subtraction:  out = 4 * image - 4 * blurred_image + 128

    The blurred image captures slow changes in brightness and color (camera
    lighting, pigmentation). Subtracting it leaves the fine details, like
    tiny red dots (microaneurysms) and bright spots (exudates), on a neutral
    gray background. We then gray out everything outside the eye circle,
    where the blur creates a bright ring artifact.
    """
    size = img.size[0]
    blurred = img.filter(ImageFilter.GaussianBlur(radius=size / 30))
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(blurred, dtype=np.float32)
    out = np.clip(4 * a - 4 * b + 128, 0, 255)

    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    outside = (yy - center) ** 2 + (xx - center) ** 2 > (0.9 * size / 2) ** 2
    out[outside] = 128
    return Image.fromarray(out.astype(np.uint8))


def process_one(job):
    src_path, dst_path, size, do_enhance = job
    try:
        img = Image.open(src_path).convert("RGB")
        img = pad_to_square(crop_black_border(img))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        if do_enhance:
            img = enhance(img)
        img.save(dst_path, quality=95)
        return None
    except Exception as e:                       # some dataset images are corrupt
        return f"{src_path}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder of raw images")
    p.add_argument("--dst", required=True, help="output folder")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--enhance", action="store_true", help="apply local average color subtraction")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    exts = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
    jobs = []
    for name in sorted(os.listdir(args.src)):
        if name.lower().endswith(exts):
            stem = os.path.splitext(name)[0]
            jobs.append((os.path.join(args.src, name),
                         os.path.join(args.dst, stem + ".jpg"),
                         args.size, args.enhance))

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


if __name__ == "__main__":
    main()
