"""
3D eye view: the heat map, wrapped onto the back wall of an eyeball you can turn.

heatmap.py makes flat pictures. This script makes ONE web page per run
(<run>/eye3d/viewer.html, opens in any browser, needs no internet) in which each
chosen validation eye is drawn as a 3D eyeball: the photo sits on the inside back
wall (where the retina is), the model's evidence is painted on it in shades of
red, and you can turn, zoom and click to find the exact spot. It adds three
things heatmap.py does not have:

  1. A SHARPER heat map. Grad-CAM (heatmap.py) reads the last stage, where one
     cell covers 32x32 pixels: it says which REGION mattered, not which spot.
     Here that region map is multiplied by a fine map from an earlier stage
     (one cell = 8x8 pixels), computed with LayerCAM (Jiang et al., 2021):
     Grad-CAM gives each feature map ONE weight for the whole picture, which
     only works deep in the network; LayerCAM gives every position its own
     weight (the positive gradient right there), which also works early, where
     the maps are still detailed. The product keeps "the right region" from
     the deep map and "which pixels inside it" from the early one.
  2. A VESSEL sketch: thin dark lines found by a classical filter (darker than
     the local average, long and thin). It is NOT a trained vessel model: it
     misses faint vessels and can pick up long hemorrhages. Its only job is to
     answer "does this hot spot sit on a vessel-like line?".
  3. A list of FINDINGS: each connected hot region gets a number, its position
     as on a clock face, its size, and a plain description of what the pixels
     under it look like (on a vessel-like line / a dark spot / a bright spot).
     That description is about pixels. It is not a diagnosis.

WHAT THE 3D IS, HONESTLY: one fundus photo is flat. It holds no depth, so
nothing here measures the eye's real shape, swelling or thickness (that needs an
OCT scan, which this project does not have). The page assumes a standard round
eyeball and that the photo covers a cap of about --cap-deg degrees around the
centre of the back wall (about right for a 45-degree fundus camera; you can change
it in the page). So positions ACROSS the retina are roughly right, depth is a
drawing. "Raise the hot spots" in the page lifts red areas toward you so they
are easier to find; it does not mean anything is raised in the eye. And the
model only knows diabetic retinopathy: it cannot tell an infection, or any
other disease, from what it was trained on.

Only VALIDATION eyes are used (the test set stays untouched), as in heatmap.py.
Usage:
  python eye3d.py --run runs\\exp15 --images %USERPROFILE%\\retina-data\\processed_512_enhanced_tight
  python eye3d.py --run runs\\exp15 --images ... --per-group 6
  python eye3d.py --run runs\\exp15 --images ... --image 12345_left 9464_right
Then open <run>\\eye3d\\viewer.html. findings.json next to it holds the same
numbers as a table, and heat\\<name>.png the heat maps as gray pictures (for
oct3d.py --heat). The page embeds the chosen photos, so, like the photos
themselves, never commit it to git.
"""

import argparse
import base64
import io
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from data import make_transforms
from heatmap import REFERABLE_FROM, choose_eyes
from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "eye3d_viewer.html")
DATA_MARK = "/*EYE_DATA*/"        # the spot in the template where the eyes are written


# ---------------------------------------------------------------------------
# 1. The heat map: Grad-CAM for the region, LayerCAM for the detail
# ---------------------------------------------------------------------------

class MultiScaleCAM:
    """
    Watches TWO places in the network during one forward + backward pass:
      - the last stage (16x16 cells at 512 px): Grad-CAM, exactly as heatmap.py;
      - an earlier stage (`fine_stage`, counted 1-4; stage 2 = 64x64 cells): LayerCAM.
    fine_stage 0 switches the fine map off (the result is then heatmap.py's map).
    """

    def __init__(self, model, fine_stage=2):
        self.model = model
        self.fine_stage = fine_stage
        self.saved = {}                           # "coarse"/"fine" -> {"act": ..., "grad": ...}
        model.stages[-1].register_forward_hook(self._hook("coarse"))
        if fine_stage:
            model.stages[fine_stage - 1].register_forward_hook(self._hook("fine"))

    def _hook(self, key):
        # A forward hook runs when the stage produces its output. We keep the
        # output, and ask PyTorch to hand us its gradient during backward().
        def save(module, inputs, output):
            self.saved[key] = {"act": output}
            output.register_hook(lambda grad: self.saved[key].__setitem__("grad", grad))
        return save

    def __call__(self, x):
        """x: one image (1, 3, H, W). Returns (heat HxW in 0..1, coarse HxW in 0..1, peak, score).
        `coarse` is heatmap.py's Grad-CAM map; `heat` is the sharpened one; `peak` is the
        Grad-CAM strength before scaling (comparable between eyes, as in heatmap.py)."""
        self.model.zero_grad(set_to_none=True)
        score = self.model(x)[0]
        score.backward()
        size = x.shape[-2:]

        # Grad-CAM: one weight per feature map (its average gradient), weighted sum, positive part.
        acts, grads = self.saved["coarse"]["act"][0], self.saved["coarse"]["grad"][0]
        weights = grads.mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
        peak = cam.max().item()
        cam = cam / (peak + 1e-8)
        coarse = F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        if not self.fine_stage:
            c = coarse.detach().cpu().numpy()
            return c, c, peak, score.item()

        # LayerCAM: every POSITION of every feature map gets its own weight, the positive
        # part of the gradient there ("would more of this pattern, right here, raise the score?").
        acts, grads = self.saved["fine"]["act"][0], self.saved["fine"]["grad"][0]
        fine = F.relu((F.relu(grads) * acts).sum(dim=0))
        fine = fine / (fine.max() + 1e-8)
        fine = F.interpolate(fine[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]

        # Geometric mean: hot only where BOTH agree. The deep map vetoes detail in regions
        # that did not matter; the early map says which pixels inside a hot region carried it.
        heat = torch.sqrt(coarse * fine)
        heat = heat / (heat.max() + 1e-8)
        return heat.detach().cpu().numpy(), coarse.detach().cpu().numpy(), peak, score.item()


# ---------------------------------------------------------------------------
# 2. Classical image helpers, hand-written (numpy + a Gaussian blur)
# ---------------------------------------------------------------------------

def label_regions(mask):
    """
    Connected regions of a True/False picture. Returns (labels, count): labels has 0 for
    False pixels and 1..count for the regions; pixels touching by a side or a corner share
    a region. Method: flood fill. Take an unlabeled True pixel, give it a new number, and
    keep spreading that number to its True neighbours until none are left.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    todo = mask.copy()
    count = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if not todo[y0, x0]:
            continue                              # already reached from an earlier pixel
        count += 1
        stack = [(int(y0), int(x0))]
        todo[y0, x0] = False
        while stack:
            y, x = stack.pop()
            labels[y, x] = count
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < h and 0 <= nx < w and todo[ny, nx]:
                        todo[ny, nx] = False
                        stack.append((ny, nx))
    return labels, count


def eye_area(rgb):
    """
    True where the picture shows eye, False for the background around it (black, or the
    flat gray of --enhance). A pixel counts as eye if it differs from the corner colour;
    then each row and each column is filled between its first and last such pixel,
    because inside an enhanced eye many pixels are the same gray as the background.
    """
    corner = np.median(np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]).astype(np.float32), axis=0)
    differs = (np.abs(rgb.astype(np.float32) - corner).max(axis=-1) > 10)
    rows = np.maximum.accumulate(differs, axis=1) & np.maximum.accumulate(differs[:, ::-1], axis=1)[:, ::-1]
    cols = np.maximum.accumulate(differs, axis=0) & np.maximum.accumulate(differs[::-1], axis=0)[::-1]
    return rows & cols


def blurred(channel, radius):
    """Gaussian blur of one 2-D float array (values 0..255), via Pillow."""
    img = Image.fromarray(np.clip(channel, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)


def vessel_sketch(rgb, eye):
    """
    A rough map of vessel-like pixels: thin dark lines. Classical, no learning:
      1. Vessels are darkest in the GREEN channel (blood absorbs green light).
      2. "Darker than the neighbourhood": blurred green minus green, kept where it is
         clearly positive (more than 1 standard deviation above the eye's average).
      3. Of the connected dark regions, keep the long thin ones. For each region the
         length is measured along its main direction (4 standard deviations of its
         pixels' positions along that axis). A round blob fills ~0.8 of length^2,
         a line much less; we keep regions under 0.25 that are at least 1/10 of
         the picture long. Round dark spots (hemorrhages) are thereby left out.
    Limits: misses faint and very thin vessels; a long streak of blood can pass as a vessel,
    and so can the dark ring that --enhance leaves around a very bright spot.
    """
    size = rgb.shape[0]
    green = rgb[..., 1].astype(np.float32)
    dark = blurred(green, max(2.0, size / 50)) - green
    inside = dark[eye]
    if inside.size == 0:
        return np.zeros(eye.shape, dtype=bool)
    candidates = (dark > inside.mean() + inside.std()) & eye
    labels, count = label_regions(candidates)
    keep = np.zeros(count + 1, dtype=bool)
    ys, xs = np.nonzero(labels)
    order = np.argsort(labels[ys, xs], kind="stable")
    ys, xs, ids = ys[order], xs[order], labels[ys, xs][order]
    starts = np.searchsorted(ids, np.arange(1, count + 2))      # where each region's pixels begin
    for k in range(1, count + 1):
        py, px = ys[starts[k - 1]:starts[k]], xs[starts[k - 1]:starts[k]]
        if len(py) < 8:
            continue
        # Main direction = the direction in which the pixels spread most: the largest
        # eigenvalue of the 2x2 covariance of their positions is the variance along it.
        spread = np.linalg.eigvalsh(np.cov(np.stack([px, py]).astype(np.float64)))[-1]
        length = 4 * math.sqrt(max(spread, 0.0))
        keep[k] = length >= size / 10 and len(py) / (length ** 2 + 1e-8) < 0.25
    return keep[labels]


def find_findings(heat, rgb, eye, vessels, hot=0.5, max_findings=8, min_area=0.0005):
    """
    Turn the heat map into a short list of numbered findings: the connected regions
    where heat >= `hot` (a share of this eye's strongest spot), largest "amount of
    heat" first. Returns (findings, label picture: 0 = none, k = finding number k).
    For each finding we describe the PIXELS under it, with simple filters:
      - share on vessel-like lines (vessel_sketch);
      - tone: brightness there compared with the local average, in units of the
        eye's own variation (below -0.5: darker, above +0.5: brighter);
      - share inside the eye (heat OUTSIDE the eye is a warning: the model should
        not find evidence in the border).
    """
    size = heat.shape[0]
    labels, count = label_regions(heat >= hot)
    gray = rgb.astype(np.float32).mean(axis=-1)
    contrast = gray - blurred(gray, max(2.0, size / 20))
    scale = contrast[eye].std() + 1e-6 if eye.any() else 1.0
    centre, radius = (size - 1) / 2, size / 2

    found = []
    for k in range(1, count + 1):
        region = labels == k
        area = region.sum() / region.size
        if area < min_area:
            continue
        ys, xs = np.nonzero(region)
        w = heat[region]
        cy, cx = float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())   # heat-weighted centre
        dx, dy = (cx - centre) / radius, (centre - cy) / radius                    # -1..1, y up
        tone = float((contrast[region] * w).sum() / w.sum() / scale)
        on_vessel = float(vessels[region].mean())
        in_eye = float(eye[region].mean())
        if in_eye < 0.5:
            looks = "mostly OUTSIDE the eye: a warning sign, the model should not find evidence there"
        elif on_vessel >= 0.3:
            looks = "largely on a vessel-like dark line"
        elif tone < -0.5:
            looks = "a spot darker than its surroundings (as blood spots are)"
        elif tone > 0.5:
            looks = "a spot brighter than its surroundings (as fatty deposits are)"
        else:
            looks = "nothing the simple filters can name"
        # Clock position as seen in the photo: 12 = top, 3 = right.
        hour = int(round((math.degrees(math.atan2(dx, dy)) % 360) / 30)) % 12 or 12
        found.append({
            "x": dx, "y": dy,
            "radius": math.sqrt(area * 4 / math.pi),      # radius of a circle of equal area, photo radius = 1
            "area_pct": 100 * float(area),
            "peak": float(w.max()), "mean": float(w.mean()),
            "from_centre_pct": 100 * math.hypot(dx, dy), "clock": hour,
            "on_vessel_pct": 100 * on_vessel, "in_eye_pct": 100 * in_eye, "tone": tone,
            "looks": looks, "_mass": float(w.sum()), "_label": k,
        })
    found.sort(key=lambda f: -f["_mass"])
    found = found[:max_findings]
    numbered = np.zeros(labels.shape, dtype=np.uint8)
    for number, f in enumerate(found, start=1):
        numbered[labels == f.pop("_label")] = number
        f.pop("_mass")
        f["id"] = number
    return found, numbered


# ---------------------------------------------------------------------------
# 3. Packing pictures into the page
# ---------------------------------------------------------------------------

def to_data_url(array, fmt, size=None, smooth=True):
    """A numpy picture -> 'data:image/...;base64,...' text that a web page can show directly."""
    img = Image.fromarray(array)
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 88} if fmt == "JPEG" else {"optimize": True}))
    kind = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_eye(cam, eval_tf, image_path, size, device, hot, max_findings):
    """Everything the page needs for one eye (heat, vessel sketch, findings), as a dict."""
    img = Image.open(image_path).convert("RGB").resize((size, size))
    x = eval_tf(img).unsqueeze(0).to(device)
    heat, coarse, peak, score = cam(x)
    rgb = np.asarray(img)
    eye = eye_area(rgb)
    vessels = vessel_sketch(rgb, eye)
    findings, numbered = find_findings(heat, rgb, eye, vessels, hot=hot, max_findings=max_findings)
    small = min(size, 256)                        # heat is smooth: 256 px is plenty and keeps the page small
    return {
        "size": size, "score": score, "peak": peak,
        "heat_outside_eye_pct": 100 * float(heat[~eye].sum() / (heat.sum() + 1e-8)),
        "vessel_pct": 100 * float(vessels[eye].mean()) if eye.any() else 0.0,
        "findings": findings,
        "photo": to_data_url(rgb, "JPEG"),
        "heat": to_data_url((heat * 255).round().astype(np.uint8), "PNG", small),
        "heat_full": to_data_url((heat * 255).round().astype(np.uint8), "PNG"),      # full size, saved as a file by main()
        "coarse": to_data_url((coarse * 255).round().astype(np.uint8), "PNG", small),
        "vessels": to_data_url(vessels.astype(np.uint8) * 255, "PNG"),
        "labels": to_data_url(numbered, "PNG", small, smooth=False),
    }


def write_viewer(eyes, meta, out_dir):
    """Fill the page template with the eyes; also write the numbers as findings.json."""
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    if DATA_MARK not in page:
        raise SystemExit(f"{TEMPLATE} has no {DATA_MARK} marker.")
    # "</" must not appear inside a <script> block; "<\/" means the same to JavaScript.
    payload = json.dumps({"meta": meta, "eyes": eyes}).replace("</", "<\\/")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page.replace(DATA_MARK, payload))
    pictures = ("photo", "heat", "coarse", "vessels", "labels")
    with open(os.path.join(out_dir, "findings.json"), "w") as f:
        json.dump({"meta": meta, "eyes": [{k: v for k, v in e.items() if k not in pictures} for e in eyes]},
                  f, indent=2)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a finished run folder, e.g. runs/exp15")
    p.add_argument("--images", required=True, help="the preprocessed images that run trained on")
    p.add_argument("--per-group", type=int, default=3,
                   help="eyes per group (missed, false alarm, caught, cleared)")
    p.add_argument("--image", nargs="*", default=[], help="specific validation image names instead")
    p.add_argument("--fine-stage", type=int, default=2, choices=[0, 1, 2, 3],
                   help="stage whose detail sharpens the map (2 = 8x8-pixel cells); 0 = plain Grad-CAM, as heatmap.py")
    p.add_argument("--hot", type=float, default=0.5,
                   help="a finding = a connected region with heat at least this share of the eye's strongest spot")
    p.add_argument("--max-findings", type=int, default=8)
    p.add_argument("--cap-deg", type=float, default=36.0,
                   help="assumed half-angle of the retina the photo covers, seen from the eyeball's centre "
                        "(~36 for a 45-degree camera in a 24 mm eye; adjustable in the page)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})                   # older runs: the default ResNet-18 with avg pooling
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                  # dropout off, BatchNorm uses its saved statistics
    cam = MultiScaleCAM(model, args.fine_stage)
    size = ckpt["size"]
    _, eval_tf = make_transforms(size, ckpt["mean"], ckpt["std"])

    # Which eyes: the same rules as heatmap.py (validation only, test names refused).
    val = pd.read_csv(os.path.join(args.run, "val_predictions.csv"), converters={"score": float})
    val["image"] = val["image"].astype(str)
    if "referred" not in val:
        val["referred"] = val["score"] >= ckpt["threshold"]
    test_split = os.path.join(args.run, "test_split.csv")
    test_names = set(pd.read_csv(test_split)["image"].astype(str)) if os.path.exists(test_split) else set()
    if args.image:
        bad = [n for n in args.image if n in test_names]
        if bad:
            raise SystemExit(f"{bad} are TEST images. The test set stays untouched: pick validation images.")
        chosen = [("chosen", row) for _, row in val[val["image"].isin(args.image)].iterrows()]
        missing = set(args.image) - set(val["image"])
        if missing:
            print(f"Not in this run's validation set, skipped: {sorted(missing)}")
    else:
        chosen = choose_eyes(val, args.per_group, args.seed)
    if not chosen:
        raise SystemExit("No validation eyes to show.")

    print(f"Model from epoch {ckpt['epoch']} (val AUC {ckpt['val_auc']:.3f}); {len(chosen)} validation eyes")
    eyes = []
    for group, row in chosen:
        name = row["image"]
        eye = build_eye(cam, eval_tf, os.path.join(args.images, f"{name}.jpg"), size, device,
                        args.hot, args.max_findings)
        eye.update({"name": name, "group": group, "grade": int(row["level"]),
                    "sick": bool(row["level"] >= REFERABLE_FROM), "referred": bool(row["referred"])})
        eyes.append(eye)
        top = eye["findings"][0]["looks"] if eye["findings"] else "-"
        print(f"  {group:12s} {name:14s} grade {eye['grade']}  score {eye['score']:+.2f}  "
              f"peak {eye['peak']:.2f}  {len(eye['findings'])} findings; strongest: {top}")

    stride = 2 ** (args.fine_stage + 1) if args.fine_stage else 32
    meta = {"run": os.path.basename(os.path.normpath(args.run)), "cap_deg": args.cap_deg, "hot": args.hot,
            "fine_stage": args.fine_stage, "cell_px": stride, "coarse_cell_px": 32,
            "epoch": int(ckpt["epoch"]), "val_auc": float(ckpt["val_auc"])}
    # The heat maps also as plain gray pictures (photo frame, 255 = this eye's strongest spot):
    # oct3d.py --heat takes one to lay it onto a measured OCT surface.
    heat_dir = os.path.join(args.run, "eye3d", "heat")
    os.makedirs(heat_dir, exist_ok=True)
    for eye in eyes:
        with open(os.path.join(heat_dir, f"{eye['name']}.png"), "wb") as f:
            f.write(base64.b64decode(eye["heat_full"].split(",", 1)[1]))
        del eye["heat_full"]
    path = write_viewer(eyes, meta, os.path.join(args.run, "eye3d"))
    print(f"Open in a browser: {path}  ({os.path.getsize(path) / 1e6:.1f} MB; it contains the photos: do not commit it)")


if __name__ == "__main__":
    main()
