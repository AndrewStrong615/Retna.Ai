"""
Step 3: Loading labels and images, splitting the data, and augmentation.

Expects a CSV with two columns, like the EyePACS trainLabels.csv:
    image,level
    10_left,0
    10_right,0
    13_left,2
where `level` is the diabetic retinopathy grade from 0 (none) to 4 (proliferative).
"""

import io
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_labels(csv_path, referable_from=2):
    """
    Turn the 0-4 grade into a yes/no question: does this eye need a specialist?
    "Referable" diabetic retinopathy is conventionally grade 2 (moderate) or worse.
    """
    df = pd.read_csv(csv_path)
    df["label"] = (df["level"] >= referable_from).astype(int)
    # EyePACS names images like "10_left" and "10_right": the number is the patient.
    df["patient"] = df["image"].astype(str).str.split("_").str[0]
    return df


def split_by_patient(df, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split by PATIENT, not by image.

    A person's left and right eyes look very similar. If one eye lands in
    training and the other in testing, the model gets a sneak peek, and your
    test score comes out higher than it really is. This mistake (called
    data leakage) shows up in published papers, so guard against it from day one.
    """
    patients = sorted(df["patient"].unique())
    random.Random(seed).shuffle(patients)
    n_test = int(len(patients) * test_frac)
    n_val = int(len(patients) * val_frac)
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])

    def which(p):
        return "test" if p in test_p else "val" if p in val_p else "train"

    split = df["patient"].map(which)
    return df[split == "train"], df[split == "val"], df[split == "test"]


def compute_mean_std(df, image_dir, n=1000, seed=0):
    """Average color and spread of YOUR images, used to normalize inputs."""
    rows = df.sample(min(n, len(df)), random_state=seed)
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    for name in rows["image"]:
        a = np.asarray(Image.open(f"{image_dir}/{name}.jpg").convert("RGB"),
                       dtype=np.float64) / 255.0
        total += a.sum(axis=(0, 1))
        total_sq += (a ** 2).sum(axis=(0, 1))
        count += a.shape[0] * a.shape[1]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return mean.tolist(), std.tolist()


def estimate_background(df, image_dir, n=50, seed=0):
    """
    The color outside the eye: black after plain preprocessing, gray (128)
    after --enhance. Rotation augmentation uncovers new corners, and they must
    be filled with this same color, or training photos would get corners that
    validation and test photos never have.

    We read it from the four corners of a few training images. JPEG
    compression makes each pixel slightly off, so we take the median.
    """
    rows = df.sample(min(n, len(df)), random_state=seed)
    corners = []
    for name in rows["image"]:
        a = np.asarray(Image.open(f"{image_dir}/{name}.jpg").convert("RGB"))
        corners += [a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]
    return tuple(int(v) for v in np.median(corners, axis=0).round())


def _chance(p):
    """True with probability p. Uses torch's random numbers (not Python's), like the
    torchvision transforms, so every loader worker gets its own seeded stream."""
    return torch.rand(1).item() < p


def _uniform(low, high):
    return low + (high - low) * torch.rand(1).item()


class RandomGamma:
    """
    Another camera's exposure curve: every pixel value v (0 to 1) becomes v ** gamma.
    gamma < 1 lifts the dark parts (a brighter-looking photo), gamma > 1 darkens them.
    Written by hand: torchvision has adjust_gamma but no random version of it.
    """

    def __init__(self, low, high, p):
        self.low, self.high, self.p = low, high, p

    def __call__(self, img):
        if not _chance(self.p):
            return img
        return transforms.functional.adjust_gamma(img, _uniform(self.low, self.high))


class RandomDownUp:
    """
    A lower-resolution camera: shrink the photo to a random 50-100% of its size, then
    stretch it back. Fine detail (tiny microaneurysms) is lost the way a cheaper
    sensor would lose it, while the photo keeps the size the network expects.
    """

    def __init__(self, low, high, p):
        self.low, self.high, self.p = low, high, p

    def __call__(self, img):
        if not _chance(self.p):
            return img
        w, h = img.size
        f = _uniform(self.low, self.high)
        small = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomJPEG:
    """
    JPEG compression at a random quality: blocky 8x8 artifacts and smeared colour, like
    the .jpg photos of other clinics (APTOS, Messidor-2) or a photo re-saved several times.
    Done for real (save to memory, read back), not imitated.
    """

    def __init__(self, low, high, p):
        self.low, self.high, self.p = low, high, p

    def __call__(self, img):
        if not _chance(self.p):
            return img
        quality = int(torch.randint(self.low, self.high + 1, (1,)).item())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def make_transforms(size, mean, std, background=(0, 0, 0), rotate_interp="bilinear", aug="standard"):
    """
    Augmentation: show the model slightly different versions of each image
    every epoch, so it learns what disease looks like instead of memorizing
    individual photos. Each change here is one that real photos vary by:
    eyes get photographed at different angles, flipped (left vs. right eye),
    and with different brightness and contrast.

    aug="strong" (exp17) adds what differs between CAMERAS, so the model learns to
    ignore it: softer optics (blur), lower resolution, a different exposure curve
    (gamma), stronger colour/contrast differences and JPEG compression. The order
    follows a camera: optics -> sensor -> colour processing -> saving the file.
    aug="standard" is exactly the old list (same steps, same random numbers drawn).

    rotate_interp="nearest" and background=(0, 0, 0) reproduce exp1/exp2.
    """
    interp = {"bilinear": transforms.InterpolationMode.BILINEAR,
              "nearest": transforms.InterpolationMode.NEAREST}[rotate_interp]
    steps = [
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Bilinear: each new pixel blends its 4 nearest neighbors. Nearest just
        # copies one, which gives tiny lesions jagged edges (but is ~1/3 faster
        # on the CPU, so it stays available as a speed fallback).
        # fill: color for the corners the rotation uncovers (see estimate_background).
        transforms.RandomRotation(180, interpolation=interp, fill=background),
    ]
    if aug == "standard":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
    elif aug == "strong":
        steps += [
            # kernel 9 covers about +-3 sigma at the largest sigma (1.5 px at 512 px).
            transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 1.5))], p=0.3),
            RandomDownUp(0.5, 1.0, p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),
            RandomGamma(0.7, 1.4, p=0.5),
            RandomJPEG(30, 90, p=0.3),
        ]
    else:
        raise ValueError(f"unknown aug {aug!r}: use 'standard' or 'strong'")
    train_tf = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([             # no randomness when measuring performance
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class FundusDataset(Dataset):
    """
    Returns (image, grade). The grade is the full 0-4 level, not the yes/no
    label: the loss function decides what to learn from it (only "grade >= 2"
    for --target binary, every grade boundary for --target ordinal).
    """

    def __init__(self, df, image_dir, transform):
        self.names = df["image"].tolist()
        self.grades = df["level"].tolist()
        # Usually every image is in image_dir. A table with an "image_dir" column
        # (train.py --extra-images) says per row which folder holds that image.
        self.dirs = df["image_dir"].tolist() if "image_dir" in df else [image_dir] * len(self.names)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(f"{self.dirs[i]}/{self.names[i]}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(self.grades[i], dtype=torch.long)
