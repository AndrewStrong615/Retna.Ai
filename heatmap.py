"""
Heat maps: WHERE in the photo did the model look when it scored an eye?

This is Grad-CAM (Selvaraju et al., 2017), written from scratch. The idea:
  1. The last stage of the network produces 512 small "feature maps" (12x12 at
     384 px). Each one lights up where it found some pattern it has learned,
     for example "small dark red dot" or "bright yellow spot".
  2. We ask: if this feature map got a little stronger, how much would the
     eye's disease score go up? That is the gradient of the score with respect
     to the map, and backpropagation computes it for free.
  3. Maps that push the score up get a positive weight. We add up all 512 maps
     with those weights and keep only the positive part: the places whose
     evidence made the model think "more disease".
  4. We stretch the 12x12 result to the photo's size and color it: red = strong
     evidence for disease, blue = little.

What to look for: on sick eyes, the hot spots should sit on lesions (red dots,
blots, yellow exudates), not on the black/gray border, the optic disc only, or
the edge of the photo. Mistakes are the most interesting: what did it see in a
healthy eye it flagged, and what did it overlook in a sick eye it missed?

Reading the colors: each map is scaled to ITS OWN strongest spot, so every
picture shows some red. The "peak" number in the label says how strong that
strongest spot really was; a healthy eye with a low peak has little evidence anywhere.

Limits: one cell of the 12x12 map covers 32x32 pixels, so this shows regions,
not single microaneurysms. And a heat map shows what the model USED, which is
not proof that it is right.

Only VALIDATION eyes are used (the test set stays untouched, per the project
rules). Usage:
  python heatmap.py --run runs\\exp4 --images %USERPROFILE%\\retina-data\\processed_384_enhanced
  python heatmap.py --run runs\\exp4 --images ... --per-group 8
  python heatmap.py --run runs\\exp4 --images ... --image 12345_left
Pictures are written to <run>\\heatmaps\\ .
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data import make_transforms
from model import build_model

REFERABLE_FROM = 2


def colorize(heat):
    """
    Turn values in 0..1 into colors, blue (low) -> green -> yellow -> red (high),
    like the common "jet" color map. Hand-written so no plotting library is needed.
    """
    h = np.clip(heat, 0, 1)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class GradCAM:
    """
    Grad-CAM for our RetinaResNet. It watches the output of the LAST stage
    (model.stages[-1]), because that is the last place where the network still
    keeps a spatial map; after it, average pooling throws "where" away.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        # A forward hook runs every time the stage produces its output. We keep
        # the output, and ask PyTorch to also hand us its gradient later.
        model.stages[-1].register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x):
        """x: one image tensor (1, 3, H, W). Returns (heat map HxW in 0..1, peak, score).
        The map is scaled so its strongest spot is 1, so EVERY image shows some red,
        even a healthy one with almost no evidence. `peak` is the strength before
        scaling: compare it between images to see how strong the evidence really is."""
        self.model.zero_grad(set_to_none=True)
        score = self.model(x)[0]                 # the eye's disease score (one number)
        score.backward()                         # fills self.gradients
        acts = self.activations[0]               # (512, h, w): what each feature map found, where
        grads = self.gradients[0]                # (512, h, w): how much each spot pushes the score
        weights = grads.mean(dim=(1, 2))         # one weight per feature map
        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))   # keep only evidence FOR disease
        peak = float(cam.max())
        cam = cam / (peak + 1e-8)                # scale to 0..1 for this image
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), peak, score.item()


def choose_eyes(val, per_group, seed):
    """
    Pick validation eyes from four groups. For the two kinds of mistakes, take
    the most confident ones (they teach the most); for correct ones, a random
    sample. Returns a list of (group name, row).
    """
    sick = val["level"] >= REFERABLE_FROM
    groups = [
        ("missed", val[sick & ~val["referred"]].sort_values("score").head(per_group)),
        ("false_alarm", val[~sick & val["referred"]].sort_values("score", ascending=False).head(per_group)),
        ("caught", val[sick & val["referred"]]),
        ("cleared", val[~sick & ~val["referred"]]),
    ]
    chosen = []
    for name, part in groups:
        if name in ("caught", "cleared"):
            part = part.sample(min(per_group, len(part)), random_state=seed)
        chosen += [(name, row) for _, row in part.iterrows()]
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a finished run folder, e.g. runs/exp4")
    p.add_argument("--images", required=True, help="the preprocessed images that run trained on")
    p.add_argument("--per-group", type=int, default=4,
                   help="eyes per group (missed, false alarm, caught, cleared)")
    p.add_argument("--image", nargs="*", default=[], help="specific validation image names instead")
    p.add_argument("--alpha", type=float, default=0.45, help="how strongly to overlay the colors")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    arch = ckpt.get("arch", {})                   # older runs: the default ResNet-18 with avg pooling
    model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                        dropout=arch.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                  # dropout off, BatchNorm uses its saved statistics
    cam = GradCAM(model)
    _, eval_tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])

    val = pd.read_csv(os.path.join(args.run, "val_predictions.csv"), converters={"score": float})
    if "referred" not in val:                     # older runs: recompute the decision
        val["referred"] = val["score"] >= ckpt["threshold"]
    test_names = set()
    test_split = os.path.join(args.run, "test_split.csv")
    if os.path.exists(test_split):
        test_names = set(pd.read_csv(test_split)["image"].astype(str))

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

    out_dir = os.path.join(args.run, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Model from epoch {ckpt['epoch']} (val AUC {ckpt['val_auc']:.3f}); "
          f"{len(chosen)} validation eyes -> {out_dir}")
    for group, row in chosen:
        name = str(row["image"])
        img = Image.open(os.path.join(args.images, f"{name}.jpg")).convert("RGB")
        img = img.resize((ckpt["size"], ckpt["size"]))
        x = eval_tf(img).unsqueeze(0).to(device)
        heat, peak, score = cam(x)

        photo = np.asarray(img, dtype=np.float32)
        overlay = (1 - args.alpha) * photo + args.alpha * colorize(heat).astype(np.float32)
        side = Image.new("RGB", (2 * ckpt["size"], ckpt["size"] + 24), (255, 255, 255))
        side.paste(img, (0, 24))
        side.paste(Image.fromarray(overlay.clip(0, 255).astype(np.uint8)), (ckpt["size"], 24))
        verdict = "REFERRED" if row["referred"] else "not referred"
        ImageDraw.Draw(side).text(
            (4, 5), f"{group}: {name} | grade {int(row['level'])} | score {score:+.2f} | {verdict} | peak {peak:.2f}",
            fill=(0, 0, 0))
        side.save(os.path.join(out_dir, f"{group}_{name}.png"))
        print(f"  {group:12s} {name:14s} grade {int(row['level'])}  score {score:+.2f}  "
              f"{verdict:12s}  peak evidence {peak:.2f}")


if __name__ == "__main__":
    main()
