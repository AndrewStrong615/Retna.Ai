"""
photo2oct.py: a ROUGH ESTIMATE of what an OCT scan of the macula would show, from a
fundus photo alone.

READ THIS FIRST: what can and cannot work
-----------------------------------------
A fundus photo is a flat picture. It holds NO depth. An OCT scan measures depth. So
nothing here "sees" the retina's thickness in the photo; it can only GUESS it from
things that tend to go with it. There are two guessers, and they are very different:

  1. The ATLAS (works today, nothing trained). It finds two landmarks in the photo, the
     optic disc (brightest blob) and the fovea (darkest smooth spot beside it), and lays
     the thickness map of an AVERAGE HEALTHY eye around that fovea: a pit in the middle,
     a thicker ring around it, a little thicker toward the disc. The photo decides only
     WHERE the map goes and how big it is. It is the same map for every eye, so it can
     never show swelling, thinning or anything else about THIS eye. It answers "what
     would a normal scan look like here?", not "what does this eye's scan look like?".

  2. The NETWORK (ThicknessNet, written here from scratch, starts from random weights).
     It looks at the macula in the photo and predicts how THIS eye DIFFERS from the atlas
     (e.g. "thicker here, where there are yellow spots"). That needs training on PAIRS:
     photo + real OCT of the same eye. The only real pairs so far: OLIVES (convert_olives.py),
     3,104 NEAR-INFRARED pictures of 87 patients. Trained on them (2026-11-25) the network
     was about as good as answering the average map (held-out error 40.4 um vs 40.5): 61
     training patients are far too few. And a colour photo is not an infrared picture.
     A network trained on cartoons (`synthetic-pairs`) says nothing about a real photo.
     Published work with ~97,000 real pairs (Holmberg et al. 2020, infrared fundus
     pictures) reached an average error of ~33 um, and worse where it matters (50-60 um
     on thick retinas): "rough" is the right word even with plenty of data.

Either way the result is an ESTIMATE. The page and every file say so. It is not a
measurement and must never be read as one.

What `estimate` writes
----------------------
  <out>/oct.png         THE OUTPUT: one OCT-style picture, the estimated B-scan through the
                        fovea. It is DRAWN from the estimated thickness (see below), with a
                        thin line "ESTIMATED ... not a scan" under it (--no-caption removes
                        the line; the same words always stay inside the file's text fields).
  <out>/estimate.json, thickness_um.npy   the numbers behind the picture.
  Only with --page:
  <out>/viewer.html     the oct3d.py page, marked ESTIMATED: the photo curved onto the eye
                        wall, the estimated surface over a 6 x 6 mm square at the fovea.
                        The B-scans in it are DRAWN from the estimate (layers painted at
                        fixed fractions of the thickness, vessel shadows under the photo's
                        vessels): cartoons of a scan, not a scan.
  <out>/landmarks.png   the photo with disc, fovea and the square drawn in (check it by eye!)

Assumptions (all written into estimate.json):
  - disc centre to fovea = 4.5 mm (it varies by roughly +-0.3 mm between people): this
    sets every millimetre in the result;
  - the atlas numbers are approximate published averages for healthy adults (Spectralis
    OCT, whole retina; e.g. Grover et al. 2010: centre ~270 um, inner ring ~335-345,
    outer ring ~290-310). They were typed in from memory of those tables, not fitted to
    data: check them against the paper before quoting any number.

Usage:
  python photo2oct.py estimate --photo eye.jpg --out runs\\p2o_eye1
  python photo2oct.py estimate --photo eye.jpg --out runs\\p2o_eye1 --model runs\\p2o_net\\model.pt
  python photo2oct.py synthetic-pairs --out pairs_demo --eyes 300      (cartoon pairs, to test the code)
  python photo2oct.py make-pair --oct runs\\oct_eye1 --photo eye.jpg --patient P17 --nasal left --pairs pairs_real
  python photo2oct.py train --pairs pairs_demo --out runs\\p2o_net
  python photo2oct.py train --pairs pairs_olives --out runs\\p2o_olives --epochs 60 --augment
"""

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import oct3d
import register as reg
from eye3d import blurred, eye_area

FIELD_MM = 6.0          # the estimate covers a 6 x 6 mm square centred on the fovea (the usual macular scan)
DISC_FOVEA_MM = 4.5     # assumed distance from the disc's centre to the fovea
IN_PX, OUT_PX = 128, 64   # the network sees the square at 128 px and answers at 64 px
RESID_UM = 100.0        # the network's output is in units of 100 um (keeps its numbers near 1)


# ---------------------------------------------------------------------------
# 1. The atlas: the thickness map of an average healthy macula
# ---------------------------------------------------------------------------

def atlas_thickness(size=OUT_PX, nasal="left", field_mm=FIELD_MM):
    """
    Whole-retina thickness (um) of an AVERAGE healthy eye over the field_mm square centred
    on the fovea, as a size x size map. `nasal`: on which side of the picture the optic
    disc lies ("left" or "right"); the retina is a little thicker toward it.
    Shape: 225 um at the bottom of the foveal pit, rising to a ~345 um ring about 1 mm out,
    then falling slowly outward. Approximate published averages, see the file's docstring.
    """
    c = (np.arange(size) + 0.5) / size * field_mm - field_mm / 2          # mm from the fovea
    x, y = np.meshgrid(c, c)
    r = np.hypot(x, y)
    toward_disc = -x if nasal == "left" else x                           # mm toward the disc
    t = 225 + 120 * (1 - np.exp(-(r / 0.45) ** 2)) - 30 * np.maximum(r - 0.9, 0) + 6 * toward_disc * np.minimum(r, 1)
    return t.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Landmarks: where are the disc and the fovea in this photo?
# ---------------------------------------------------------------------------

def square_photo(rgb):
    """Crop to the eye's bounding box and pad to a square (black), like preprocess.py's first steps."""
    eye = eye_area(rgb)
    ys, xs = np.nonzero(eye)
    if len(ys) < 0.05 * eye.size:
        raise SystemExit("No eye found in the picture (it looks like one flat colour).")
    crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), np.uint8)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return out


def find_landmarks(rgb):
    """
    Classical, nothing learned. Returns positions as fractions of the picture (x, y in 0..1):
      disc  = the brightest blob (brightness blurred over ~1/40 of the picture);
      fovea = the darkest smooth spot (blurred over ~1/25) that lies 25-50% of the picture
              away from the disc and not much above or below it. Blood vessels are dark
              too, but thin: the wide blur removes them, the fovea's broad shadow stays.
    Also returns `contrast`: how much darker the fovea spot is than its surroundings, in
    grey levels. A small value (< 4) means there was no clear dark spot: the fovea is then
    just the darkest of nothing, and the whole estimate may sit in the wrong place.
    """
    work = 384
    small = np.asarray(Image.fromarray(rgb).resize((work, work), Image.Resampling.BILINEAR))
    eye = eye_area(small)
    inner = blurred(eye.astype(np.float32) * 255, work / 20) > 250       # the eye minus a rim (rims are often bright)
    grey = small.astype(np.float32).mean(axis=-1)
    bright = np.where(inner, blurred(grey, work / 40), -1)
    dy, dx = np.unravel_index(np.argmax(bright), bright.shape)
    smooth = blurred(grey, work / 25)
    yy, xx = np.mgrid[:work, :work]
    dist = np.hypot(xx - dx, yy - dy) / work
    allowed = inner & (dist > 0.25) & (dist < 0.50) & (np.abs(yy - dy) < 0.15 * work)
    if not allowed.any():
        raise SystemExit("No place for a fovea beside the brightest spot: is this a macula-centred fundus photo?")
    fy, fx = np.unravel_index(np.argmin(np.where(allowed, smooth, 1e9)), smooth.shape)
    around = allowed & (np.hypot(xx - fx, yy - fy) > 0.06 * work) & (np.hypot(xx - fx, yy - fy) < 0.14 * work)
    contrast = float(smooth[around].mean() - smooth[fy, fx]) if around.any() else 0.0
    return ((dx + 0.5) / work, (dy + 0.5) / work), ((fx + 0.5) / work, (fy + 0.5) / work), contrast


def placement(disc, fovea):
    """
    From the two landmarks: how many mm the picture's width covers (disc-fovea distance =
    DISC_FOVEA_MM, assumed), on which side the disc is, and the matrix that maps a position
    in the 6 mm square (u, v in 0..1) to a position in the photo (0..1): the same kind of
    matrix register.py finds for a real scan, here simply KNOWN (no vessel matching needed,
    the square is placed on the fovea by construction).
    """
    d = float(np.hypot(fovea[0] - disc[0], fovea[1] - disc[1]))
    width_mm = DISC_FOVEA_MM / d
    k = FIELD_MM / width_mm
    matrix = [[k, 0.0, fovea[0] - k / 2], [0.0, k, fovea[1] - k / 2]]
    return width_mm, ("left" if disc[0] < fovea[0] else "right"), matrix


def landmarks_picture(rgb, disc, fovea, matrix, size=360):
    """The photo with the disc (circle), the fovea (cross) and the 6 mm square drawn in: to check by eye."""
    img = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(img)
    corners = [reg.scan_to_photo(matrix, u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    d.polygon([(x * size, y * size) for x, y in corners], outline=(90, 200, 255))
    x, y, r = disc[0] * size, disc[1] * size, 0.05 * size
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)
    x, y, r = fovea[0] * size, fovea[1] * size, 0.03 * size
    d.line([x - r, y, x + r, y], fill=(255, 255, 255), width=2)
    d.line([x, y - r, x, y + r], fill=(255, 255, 255), width=2)
    return np.asarray(img)


def photo_spots(rgb, disc, fovea, min_z=8.0, work=512):
    """
    POINTS OF INTEREST in the photo, by a classical filter (nothing learned): small, roundish
    spots that are clearly brighter or darker than their surroundings in the green channel.
      1. fine picture (blur 1.5 px) minus coarse picture (blur ~13 px): what stands out locally;
      2. measured in units of the photo's own typical variation (z; robust: median-based);
      3. connected regions with |z| > 4 are kept only if they are COMPACT (a vessel is long and
         thin: it fills little of the square around its length) and small, and their strongest
         pixel reaches `min_z`;
      4. left out: the optic disc and its surroundings, the rim of the picture, and the dark
         fovea itself.
    Returns (heat picture uint8 [work, work] with a soft blob per spot, strongest = 255; list
    of spots with kind 'bright'/'dark', position 0..1 and z).
    HONESTY: `min_z` 8 was set on ONE healthy photo (its strongest ordinary structure reached
    z 7) and on cartoon spots (z 10-14). It has not met a real diseased photo. It marks
    "something small stands out here", e.g. a hard exudate, a haemorrhage, but just as well a
    reflection, dust on the lens or a pigment spot. It is not a diagnosis, and a clean result
    does not mean a healthy eye. The disease model's own heat map (--heat) is the better source.
    """
    import math
    from eye3d import label_regions
    small = np.asarray(Image.fromarray(rgb).resize((work, work), Image.Resampling.BILINEAR))
    inner = blurred(eye_area(small).astype(np.float32) * 255, work / 20) > 250
    green = small[..., 1].astype(np.float32)
    diff = blurred(green, 1.5) - blurred(green, work / 40)
    yy, xx = np.mgrid[:work, :work]
    ok = inner & (np.hypot(xx - disc[0] * work, yy - disc[1] * work) > 0.11 * work)
    middle = np.median(diff[ok])
    z = (diff - middle) / (1.4826 * np.median(np.abs(diff[ok] - middle)) + 1e-6)
    spots, heat = [], np.zeros((work, work), np.float32)
    for sign, kind in ((1, "bright"), (-1, "dark")):
        labels, count = label_regions((sign * z > 4) & ok)
        for k in range(1, count + 1):
            py, px = np.nonzero(labels == k)
            if len(py) < 6:
                continue
            length = 4 * math.sqrt(max(np.linalg.eigvalsh(np.cov(np.stack([px, py]).astype(np.float64)))[-1], 0.0))
            peak = float((sign * z)[py, px].max())
            x, y = float(px.mean() + 0.5) / work, float(py.mean() + 0.5) / work
            at_fovea = kind == "dark" and np.hypot(x - fovea[0], y - fovea[1]) < 0.04
            if peak < min_z or length > 0.06 * work or len(py) / (length ** 2 + 1e-8) < 0.35 or at_fovea:
                continue
            spots.append({"kind": kind, "x": x, "y": y, "z": peak})
            blob = np.exp(-((xx - x * work) ** 2 + (yy - y * work) ** 2) / (2 * (0.02 * work) ** 2))
            heat = np.maximum(heat, min(1.0, peak / 16) * blob)
    if spots:
        heat = heat / heat.max()
    return (heat * 255).round().astype(np.uint8), sorted(spots, key=lambda f: -f["z"])


# ---------------------------------------------------------------------------
# 3. The network: photo of the macula -> difference from the atlas
# ---------------------------------------------------------------------------

def build_net():
    """
    ThicknessNet: a small U-Net (Ronneberger et al. 2015) from PyTorch building blocks,
    random start, ~120,000 weights.
      going down: three steps, each halves the picture (128 -> 64 -> 32 -> 16) and doubles
                  the channels, so the network sees wider and wider surroundings;
      going up:   back to 64 x 64, each step also getting the 'down' features of the same
                  size (the skip connections: they carry the fine positions the way down lost);
      output:     one number per cell = how much thicker (+) or thinner (-) than the atlas,
                  in units of 100 um.
    The last layer starts at ZERO, so an untrained network answers exactly "the atlas":
    training can only move it away from the average eye where the pairs give a reason to.
    """
    import torch
    from torch import nn

    def block(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class ThicknessNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1, self.d2, self.d3 = block(3, 16, 2), block(16, 32, 2), block(32, 64, 2)   # 64, 32, 16 px
            self.u2, self.u1 = block(64 + 32, 32), block(32 + 16, 16)
            self.out = nn.Conv2d(16, 1, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, x):                      # x: [batch, 3, 128, 128], values about -1..1
            a = self.d1(x)
            b = self.d2(a)
            c = self.d3(b)
            b = self.u2(torch.cat([self.up(c), b], dim=1))
            a = self.u1(torch.cat([self.up(b), a], dim=1))
            return self.out(a)                     # [batch, 1, 64, 64]

    return ThicknessNet()


def build_encoder_net(encoder_state=None):
    """
    The second network: the project's OWN ResNet-18 (model.py) as the part that looks, with a
    small hand-written part on top that draws the 64 x 64 correction.
      - `encoder_state`: the `encoder` weights from pretrain.py's encoder.pt (stem + stages, learned
        by this project from ~78,000 unlabeled EyePACS photos by redrawing hidden squares). They
        already know what vessels, spots and textures look like, so the few OLIVES patients only
        have to teach "which of those go with a thicker retina", not "how to see" from nothing.
        None = the same network from random weights (to measure what the pretraining adds).
      - the 128 px picture is enlarged to 256 px first, so the ResNet's 1/4, 1/8 and 1/16 size
        feature maps are 64, 32 and 16 px: the same sizes ThicknessNet works at. Stage 4 (1/32)
        is not used: whole-eye summaries matter for a disease score, less for a local map.
      - going up as in ThicknessNet (skip connections), last layer zero: untrained = the atlas.
    11.2M weights minus stage 4 = ~2.8M in the encoder, ~0.5M on top.
    """
    import torch
    from torch import nn
    from model import build_model

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class EncoderNet(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = build_model(18, "gem")
            if encoder_state is not None:
                missing, unexpected = resnet.load_state_dict(encoder_state, strict=False)
                bad = [k for k in missing if not k.startswith(("head.", "pool."))]
                if bad or unexpected:
                    raise SystemExit(f"The encoder does not fit model.py's ResNet-18: missing {bad[:4]}, unexpected {unexpected[:4]}")
            self.stem, self.s1, self.s2, self.s3 = resnet.stem, resnet.stages[0], resnet.stages[1], resnet.stages[2]
            self.u2, self.u1 = block(256 + 128, 64), block(64 + 64, 32)
            self.out = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def encoder_parameters(self):
            return [q for m in (self.stem, self.s1, self.s2, self.s3) for q in m.parameters()]

        def forward(self, x):                      # x: [batch, 3, 128, 128]
            a = self.s1(self.stem(self.up(x)))     # 64 channels, 64 px
            b = self.s2(a)                         # 128 channels, 32 px
            c = self.s3(b)                         # 256 channels, 16 px
            b = self.u2(torch.cat([self.up(c), b], dim=1))
            a = self.u1(torch.cat([self.up(b), a], dim=1))
            return self.out(a)                     # [batch, 1, 64, 64]

    return EncoderNet()


SECTOR_NAMES = ["centre", "inner disc side", "inner upper", "inner far side", "inner lower",
                "outer disc side", "outer upper", "outer far side", "outer lower"]


def etdrs_masks(size=OUT_PX, field_mm=FIELD_MM):
    """
    The 9 ETDRS sectors eye doctors read a thickness map by, as weights [9, size, size] (each sums
    to 1, so `sector_means` is a weighted average): the centre circle (1 mm across), an inner ring
    (out to 3 mm across) and an outer ring (out to 6 mm across), each ring cut into 4 quarters.
    The picture has the disc on the LEFT, so 'disc side' (nasal) is left and 'far side' (temporal) right.
    Why bother: one number per sector averages away what no photo can tell (the exact place of an
    edge, a single cyst), and keeps what a treatment decision uses (is the centre thick?).
    """
    c = (np.arange(size) + 0.5) / size * field_mm - field_mm / 2
    x, y = np.meshgrid(c, c)
    r = np.hypot(x, y)
    angle = np.degrees(np.arctan2(-y, -x)) % 360       # 0 = toward the disc (left), 90 = up, 180 = far side, 270 = down
    masks = [r < 0.5]
    for lo, hi in ((0.5, 1.5), (1.5, 3.0)):
        for quarter in range(4):
            masks.append((r >= lo) & (r < hi) & ((angle - 90 * quarter + 45) % 360 < 90))
    m = np.stack(masks).astype(np.float32)
    return m / m.sum(axis=(1, 2), keepdims=True)


def sector_means(maps, masks=None):
    """Thickness maps [n, 64, 64] (numpy) -> the 9 sector averages [n, 9]."""
    return np.einsum("khw,nhw->nk", etdrs_masks() if masks is None else masks, maps)


def to_input(crop_rgb, grey=False):
    """
    128 x 128 RGB crop (uint8, disc side on the LEFT) -> network input [3, 128, 128], about -1..1.
    grey=True (pairs without colour, e.g. OLIVES' near-infrared pictures): the three colours are
    averaged into one grey picture, which is then STANDARDIZED on its own (minus its mean, divided
    by twice its spread) and given to all 3 input channels. Standardizing removes overall
    brightness and contrast, which differ between an infrared picture and a colour photo turned
    grey, so the network sees only the pattern. It does NOT make a colour photo look infrared:
    infrared light shows deeper layers, so the same eye looks different in the two.
    """
    crop = np.asarray(crop_rgb, dtype=np.float32)
    if grey:
        g = crop.mean(axis=-1)
        g = np.clip((g - g.mean()) / (2 * g.std() + 1e-6), -2, 2)
        return np.stack([g, g, g]).astype(np.float32)
    return (crop / 127.5 - 1).transpose(2, 0, 1)


def macula_crop(rgb, matrix, nasal):
    """The 6 mm square of the photo at IN_PX, mirrored if needed so the disc side is on the LEFT
    (the network then never has to learn right eyes and left eyes separately)."""
    crop = np.clip(reg.warp_to_scan(rgb, matrix, IN_PX), 0, 255).astype(np.uint8)
    return crop[:, ::-1].copy() if nasal == "right" else crop


def predict_thickness(model_path, crop_rgb):
    """Atlas + the trained network's correction, for one crop (disc side on the left). Returns um, 64 x 64."""
    import torch
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    net = build_encoder_net() if saved.get("meta", {}).get("arch") == "resnet18" else build_net()
    net.load_state_dict(saved["model"])
    net.eval()
    with torch.no_grad():
        grey = saved.get("meta", {}).get("input") == "grey"            # trained on colourless pictures: feed grey
        resid = net(torch.from_numpy(to_input(crop_rgb, grey))[None])[0, 0].numpy() * RESID_UM
    return atlas_thickness(OUT_PX, "left") + resid, saved.get("meta", {})


# ---------------------------------------------------------------------------
# 4. Pairs: photo crop + thickness map of the same eye, the network's training data
# ---------------------------------------------------------------------------

def add_pair(folder, name, patient, crop_rgb, thick_um):
    """One pair = <name>.png (128 px crop, disc side left) + <name>.npy (64 x 64 um) + a row in pairs.csv."""
    os.makedirs(folder, exist_ok=True)
    table = os.path.join(folder, "pairs.csv")
    if os.path.exists(os.path.join(folder, name + ".png")):
        raise SystemExit(f"{folder} already holds a pair named {name}.")
    Image.fromarray(crop_rgb).save(os.path.join(folder, name + ".png"))
    np.save(os.path.join(folder, name + ".npy"), thick_um.astype(np.float32))
    new = not os.path.exists(table)
    with open(table, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "patient"])
        w.writerow([name, patient])


def synthetic_pair(seed):
    """
    A cartoon pair with a KNOWN rule, to test that the network and the training loop learn:
    the crop shows a reddish background, vessels, a dark fovea and, where the 'retina' is
    swollen (0-2 places per eye), a cluster of yellow and dark spots. The thickness is the
    atlas + those swellings + an offset of the whole map (+-15 um) that NOTHING in the
    picture reveals, like the real differences between healthy people. So a perfect
    network can remove the swelling error but not the offset error.
    Returns (crop uint8 [128, 128, 3], thickness um [64, 64], number of swellings).
    """
    rng = np.random.default_rng(7000 + seed)
    big = 256
    yy, xx = np.mgrid[:big, :big]
    mm = lambda p: (np.atleast_2d(p) + FIELD_MM / 2) / FIELD_MM * big         # mm from the fovea -> pixels
    img = np.stack([200 + rng.normal(0, 8), 105 + rng.normal(0, 8), 60 + rng.normal(0, 6)]) * np.ones((big, big, 3))
    vessels = oct3d.draw_paths(oct3d.synthetic_vessels(seed, tree=seed), mm, big, big / FIELD_MM)[..., None]
    img = img * (1 - vessels) + np.array([105.0, 28.0, 24.0]) * vessels
    fovea = np.exp(-((xx - big / 2) ** 2 + (yy - big / 2) ** 2) / (2 * (0.5 * big / FIELD_MM) ** 2))[..., None]
    img = img * (1 - 0.35 * fovea)
    c = (np.arange(OUT_PX) + 0.5) / OUT_PX * FIELD_MM - FIELD_MM / 2
    tx, ty = np.meshgrid(c, c)
    thick = atlas_thickness(OUT_PX, "left") + rng.normal(0, 15)
    count = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
    for _ in range(count):
        cx, cy = rng.uniform(-2.2, 2.2, 2)
        sigma, height = rng.uniform(0.35, 0.7), rng.uniform(80, 250)
        thick = thick + height * np.exp(-((tx - cx) ** 2 + (ty - cy) ** 2) / (2 * sigma ** 2))
        for _ in range(int(6 + height / 25)):                              # more spots where it is more swollen
            sx, sy = mm(np.array([cx, cy]) + rng.normal(0, 0.7 * sigma, 2))[0]
            spot = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.07 * big / FIELD_MM) ** 2))[..., None]
            img = img * (1 - spot) + np.array([[120.0, 20.0, 20.0], [240.0, 215.0, 110.0]][int(rng.random() < 0.5)]) * spot
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    crop = np.asarray(Image.fromarray(img).resize((IN_PX, IN_PX), Image.Resampling.LANCZOS))
    return crop, thick.astype(np.float32), count


def pairs_info(folder):
    """A pairs folder's own notes (pairs.json, written by convert_olives.py), or {} for the plain folders."""
    path = os.path.join(folder, "pairs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_pairs(folder):
    with open(os.path.join(folder, "pairs.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{folder}/pairs.csv is empty.")
    grey = pairs_info(folder).get("input") == "grey"
    x = np.stack([to_input(Image.open(os.path.join(folder, r["name"] + ".png")).convert("RGB"), grey) for r in rows])
    y = np.stack([np.load(os.path.join(folder, r["name"] + ".npy")) for r in rows])
    return x, y, [r["patient"] for r in rows]


def split_patients(patients, seed=0):
    """70 / 15 / 15% of the PATIENTS (never of the pictures: both eyes of a person stay together,
    the project's rule). Returns three boolean masks over the pairs: train, validation, test."""
    unique = sorted(set(patients))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_val = max(1, round(0.15 * len(unique)))
    val = {unique[i] for i in order[:n_val]}
    test = {unique[i] for i in order[n_val:2 * n_val]}
    p = np.array(patients)
    in_val, in_test = np.isin(p, list(val)), np.isin(p, list(test))
    return ~(in_val | in_test), in_val, in_test


def augmented(x, y, rng, max_shift=4):
    """
    One training batch seen a little differently each time, so the network cannot simply memorize the
    few hundred eyes it has (many pairs are the SAME eye at another visit). x: pictures [n, 3, 128, 128],
    y: thickness maps in um [n, 64, 64] (the thickness itself, not yet 'minus the atlas').
    Per pair, picture and map TOGETHER:
      - turned upside down with probability 1/2 (the disc stays on the left; above and below the fovea the
        retina is built alike, so an upside-down macula is still a believable macula);
      - moved by up to `max_shift` map cells (= 2 picture pixels each, about 0.4 mm at most) sideways and
        up/down; the strip that moves in is filled by mirroring the edge. The scan is never centred on the
        fovea exactly, so the network should not rely on where things sit to the pixel.
    """
    import torch
    p = max_shift

    def mirrored(t, k):                                  # k more rows/columns on every side, the edge mirrored
        t = torch.cat([t.flip(-1)[..., t.shape[-1] - k:], t, t.flip(-1)[..., :k]], dim=-1)
        return torch.cat([t.flip(-2)[..., t.shape[-2] - k:, :], t, t.flip(-2)[..., :k, :]], dim=-2)

    xp, yp = mirrored(x, 2 * p), mirrored(y, p)
    xs, ys = [], []
    for i in range(len(x)):
        dx, dy = (int(v) for v in rng.integers(-p, p + 1, 2))
        a = xp[i, :, 2 * (p + dy):2 * (p + dy) + IN_PX, 2 * (p + dx):2 * (p + dx) + IN_PX]
        b = yp[i, p + dy:p + dy + OUT_PX, p + dx:p + dx + OUT_PX]
        if rng.random() < 0.5:
            a, b = a.flip(1), b.flip(0)
        xs.append(a)
        ys.append(b)
    return torch.stack(xs), torch.stack(ys)


def train(pairs, out, epochs=30, batch=32, lr=2e-3, seed=0, log=print, augment=False, wd=1e-4, test=True,
          sector_weight=0.0, arch="unet", init=None, encoder_lr=0.1):
    """
    Train ThicknessNet on a pairs folder. Loss: mean absolute error (L1) between the
    predicted and the real thickness map, in um / 100.
    Every epoch the VALIDATION patients are scored; the best epoch's weights are kept.
    The TEST patients are scored once, at the very end (the project's rule), against two
    yardsticks that need no network at all:
      - 'atlas':    answer the average healthy map for everyone;
      - 'mean map': answer the average of the TRAINING maps for everyone.
    A network that does not clearly beat both has learned nothing from the photo.
    The same two yardsticks are also reported on the VALIDATION patients, to judge a recipe without the test.
    augment: see `augmented` (off = the plain loop, same numbers as before). wd: weight decay.
    sector_weight W (0 = off, the old loss): adds W x the mean absolute error of the 9 ETDRS sector averages
    to the loss, so the network is also paid for getting each region's AVERAGE right, not only each cell.
    The sector errors of network, atlas and mean map are reported either way.
    arch "resnet18": build_encoder_net instead of ThicknessNet; `init` = pretrain.py's encoder.pt (this
    project's own pretrained weights; none = random start). The pretrained part then learns with
    `encoder_lr` x the learning rate (0.1: it already knows a lot, big steps would wipe that out; 0 = frozen).
    test=False: do not look at the test patients at all (for trying recipes; choose on validation, then
    run the chosen recipe once with the test).
    """
    import torch
    if os.path.exists(os.path.join(out, "model.pt")):
        raise SystemExit(f"{out} already holds a trained model. Pick a new --out.")
    x, y, patients = load_pairs(pairs)
    tr, va, te = split_patients(patients, seed)
    if tr.sum() < 8 or va.sum() < 1 or te.sum() < 1:
        raise SystemExit(f"Too few pairs to split by patient ({len(set(patients))} patients).")
    atlas = atlas_thickness(OUT_PX, "left")
    resid = (y - atlas)[:, None] / RESID_UM                              # what the network must answer
    torch.manual_seed(seed)
    steps = epochs * int(np.ceil(tr.sum() / batch))
    init_info = None
    if arch == "resnet18":
        state = None
        if init:
            pre = torch.load(init, map_location="cpu", weights_only=False)
            if pre.get("arch", {}).get("depth", 18) != 18:
                raise SystemExit(f"{init} is not a ResNet-18 encoder ({pre.get('arch')}).")
            state, init_info = pre["encoder"], {"file": os.path.basename(init), "pretrain_epoch": pre.get("epoch")}
            log(f"Starting from this project's pretrained encoder {init} (pretraining epoch {pre.get('epoch')})")
        net = build_encoder_net(state)
        enc = net.encoder_parameters()
        enc_ids = {id(q) for q in enc}
        rest = [q for q in net.parameters() if id(q) not in enc_ids]
        if init and encoder_lr == 0:
            for q in enc:
                q.requires_grad_(False)                                   # frozen: only the part on top learns
            groups, max_lr = [{"params": rest}], lr
        elif init:
            groups, max_lr = [{"params": enc}, {"params": rest}], [lr * encoder_lr, lr]
        else:
            groups, max_lr = [{"params": enc + rest}], lr
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    else:
        if init:
            raise SystemExit("--init needs --arch resnet18 (ThicknessNet has no pretrained version).")
        net, max_lr = build_net(), lr
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps)   # warm-up, then a cosine fall
    masks = etdrs_masks()
    masks_t = torch.from_numpy(masks)

    def sectors_t(maps):                                                  # [n, 64, 64] tensor -> [n, 9]
        return torch.einsum("khw,nhw->nk", masks_t, maps)
    xt, rt = torch.from_numpy(x), torch.from_numpy(resid.astype(np.float32))
    yt, atlas_t = torch.from_numpy(y), torch.from_numpy(atlas)
    idx_tr = np.nonzero(tr)[0]
    rng = np.random.default_rng(seed)

    def mae(mask):                                                        # um, mean over the masked pairs
        net.eval()
        idx = np.nonzero(mask)[0]
        with torch.no_grad():                                             # in pieces: the ResNet needs more memory
            pred = np.concatenate([net(xt[idx[i:i + 128]])[:, 0].numpy() for i in range(0, len(idx), 128)]) * RESID_UM + atlas
        return float(np.abs(pred - y[mask]).mean()), pred

    best, history, best_state = 1e9, [], None
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(idx_tr)                                   # a new random order every epoch
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            if augment:
                xb, yb = augmented(xt[pick], yt[pick], rng)
                out_b, want = net(xb), ((yb - atlas_t) / RESID_UM)[:, None]
            else:
                out_b, want = net(xt[pick]), rt[pick]
            loss = (out_b - want).abs().mean()
            if sector_weight:
                loss = loss + sector_weight * (sectors_t(out_b[:, 0]) - sectors_t(want[:, 0])).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        val_mae, val_pred = mae(va)
        history.append({"epoch": epoch, "train_loss_um": float(np.mean(losses)) * RESID_UM, "val_mae_um": val_mae,
                        "val_sector_mae_um": float(np.abs(sector_means(val_pred, masks) - sector_means(y[va], masks)).mean())})
        if val_mae < best:
            best, best_state = val_mae, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            log(f"epoch {epoch:3d}: train L1 {history[-1]['train_loss_um']:.1f} um, validation error {val_mae:.1f} um")
    net.load_state_dict(best_state)
    mean_map = y[tr].mean(axis=0)

    def sector_report(mask, pred):
        """Mean error of the 9 sector averages (um), all sectors and the centre alone: network, atlas, mean map."""
        truth = sector_means(y[mask], masks)
        rows = {}
        for name, guess in (("network", pred), ("atlas", atlas[None]), ("mean_map", mean_map[None])):
            err = np.abs(sector_means(guess, masks) - truth)
            rows[name] = {"all_um": float(err.mean()), "centre_um": float(err[:, 0].mean())}
        rows["centre_correlation"] = float(np.corrcoef(sector_means(pred, masks)[:, 0], truth[:, 0])[0, 1])
        return rows

    _, val_pred = mae(va)
    val_sectors = sector_report(va, val_pred)
    results = {"arch": arch, "init": init_info, "encoder_lr": encoder_lr if init else None, "sector_weight": sector_weight,
               "val_sectors": val_sectors, "pairs": len(y), "patients": len(set(patients)), "train/val/test pairs": [int(tr.sum()), int(va.sum()), int(te.sum())],
               "augment": bool(augment), "wd": wd, "lr": lr,
               "best_val_mae_um": best, "best_epoch": int(np.argmin([h["val_mae_um"] for h in history])) + 1,
               "val_mae_atlas_um": float(np.abs(atlas - y[va]).mean()),
               "val_mae_mean_map_um": float(np.abs(mean_map - y[va]).mean())}
    log(f"VALIDATION patients: network {best:.1f} um (best epoch {results['best_epoch']}) | atlas for everyone "
        f"{results['val_mae_atlas_um']:.1f} um | mean training map {results['val_mae_mean_map_um']:.1f} um")
    log(f"  the 9 ETDRS sector averages: network {val_sectors['network']['all_um']:.1f} um | atlas "
        f"{val_sectors['atlas']['all_um']:.1f} | mean map {val_sectors['mean_map']['all_um']:.1f}; centre sector alone: "
        f"{val_sectors['network']['centre_um']:.1f} | {val_sectors['atlas']['centre_um']:.1f} | "
        f"{val_sectors['mean_map']['centre_um']:.1f} (network's centre follows the real one with correlation "
        f"{val_sectors['centre_correlation']:.2f})")
    if not test:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "results_val_only.json"), "w") as f:
            json.dump({**results, "history": history}, f, indent=2)
        log("Test patients not looked at (--no-test): no model saved; rerun the chosen recipe without --no-test.")
        return results
    test_mae, pred = mae(te)                                              # the single look at the test patients
    results.update({"test_mae_um": test_mae,
               "test_mae_atlas_um": float(np.abs(atlas - y[te]).mean()),
               "test_mae_mean_map_um": float(np.abs(mean_map - y[te]).mean()),
               "test_peak_error_um": float(np.abs(pred.max(axis=(1, 2)) - y[te].max(axis=(1, 2))).mean()),
               "test_peak_error_atlas_um": float(np.abs(atlas.max() - y[te].max(axis=(1, 2))).mean()),
               "test_sectors": sector_report(te, pred)})
    os.makedirs(out, exist_ok=True)
    notes = pairs_info(pairs)
    meta = {"pairs_folder": os.path.basename(os.path.normpath(pairs)), "epochs": epochs, "seed": seed,
            "input": notes.get("input", "colour"), "source": notes.get("source"), **results}
    ts = results["test_sectors"]
    log(f"TEST, the 9 sector averages: network {ts['network']['all_um']:.1f} um | atlas {ts['atlas']['all_um']:.1f} | mean map "
        f"{ts['mean_map']['all_um']:.1f}; centre alone: {ts['network']['centre_um']:.1f} | {ts['atlas']['centre_um']:.1f} | "
        f"{ts['mean_map']['centre_um']:.1f} (correlation {ts['centre_correlation']:.2f})")
    torch.save({"model": best_state, "meta": meta}, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({**results, "history": history}, f, indent=2)
    log(f"TEST patients (scored once): network {test_mae:.1f} um | atlas for everyone {results['test_mae_atlas_um']:.1f} um | "
        f"mean training map {results['test_mae_mean_map_um']:.1f} um")
    return results


# ---------------------------------------------------------------------------
# 5. From a thickness map to the 3D page
# ---------------------------------------------------------------------------

def drawn_volume(thick_um, vessel_mask=None, n_scans=64, width=256, depth=256, seed=0):
    """
    A DRAWING of a scan with the given thickness: a flat eye wall (a photo says nothing about
    its curve or tilt), the retina on top of it, layers painted at fixed fractions of the
    thickness (a bright nerve-fibre layer on top, two darker nuclear layers, the bright
    photoreceptor/RPE band below), shadows under the photo's vessels, speckle like real OCT.
    Only the thickness comes from the estimate; every layer inside it is a cartoon.
    Returns (volume uint8 [scan, depth, width], spacing, ILM px, lower surface px).
    """
    rng = np.random.default_rng(seed)
    spacing = {"dx": FIELD_MM * 1000 / width, "dy": FIELD_MM * 1000 / n_scans, "dz": 2000 / depth, "assumed": True}
    t = np.asarray(Image.fromarray(thick_um.astype(np.float32), mode="F").resize((width, n_scans), Image.Resampling.BILINEAR))
    low_um = np.full_like(t, 1000.0)
    ilm_um = low_um - t
    zz = ((np.arange(depth) + 0.5) * spacing["dz"])[None, :, None]
    ilm, low = ilm_um[:, None, :], low_um[:, None, :]
    frac = (zz - ilm) / (low - ilm)                                       # 0 at the inner surface, 1 at the lower one
    img = np.full((n_scans, depth, width), 14.0)
    img = np.where(frac >= 0, 105.0, img)
    img = np.where((frac >= 0) & (frac < 0.12), 175.0, img)
    img = np.where((frac >= 0.36) & (frac < 0.50), 70.0, img)
    img = np.where((frac >= 0.78) & (frac < 1.0), 60.0, img)
    img = np.where((zz >= low) & (zz < low + 45), 225.0, img)
    img = np.where(zz >= low + 45, 120.0 * np.exp(-(zz - low - 45) / 400), img)
    if vessel_mask is not None:
        m = np.asarray(Image.fromarray(vessel_mask.astype(np.float32), mode="F").resize((width, n_scans), Image.Resampling.BILINEAR))
        img = np.where(frac >= 0.12, img * (1 - 0.6 * np.clip(m, 0, 1)[:, None, :]), img)
    img = img * rng.exponential(1.0, img.shape) ** 0.5 * 1.13 + rng.normal(0, 6, img.shape)
    return (np.clip(img, 0, 255).astype(np.uint8), spacing,
            (ilm_um / spacing["dz"] - 0.5).astype(np.float32), (low_um / spacing["dz"] - 0.5).astype(np.float32))


def oct_picture(vol, caption=None, size=(768, 512)):
    """
    THE OUTPUT: one OCT-style picture, the drawn B-scan through the fovea (the middle one of
    the stack, running left to right across the 6 mm square), as an eye doctor's screen shows
    it: grey, the retina's inner surface on top, 6 mm wide and 2 mm deep stretched to 3:2.
    `caption`: a thin line of text under the picture (None = no line).
    """
    scan = Image.fromarray(vol[vol.shape[0] // 2]).resize(size, Image.Resampling.LANCZOS)
    if not caption:
        return scan
    out = Image.new("L", (size[0], size[1] + 22), 0)
    out.paste(scan, (0, 0))
    ImageDraw.Draw(out).text((8, size[1] + 5), caption, fill=170)
    return out


def estimate(photo_path, out, model=None, disc=None, fovea=None, overwrite=False, log=print, heat_path=None,
             page=False, caption=True):
    """
    Photo -> <out>/oct.png (the estimated OCT picture; that is the whole output by default),
    plus estimate.json and thickness_um.npy (the numbers behind it, for the record).
    page=True also writes the 3D page viewer.html and landmarks.png.
    """
    want_page = page
    page_path = os.path.join(out, "viewer.html")
    picture_path = os.path.join(out, "oct.png")
    if os.path.exists(picture_path) and not overwrite:
        raise SystemExit(f"{picture_path} exists. Pick a new --out or add --overwrite.")
    rgb = square_photo(np.asarray(Image.open(photo_path).convert("RGB")))
    if max(rgb.shape) > 1024:
        rgb = np.asarray(Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.LANCZOS))
    found_disc, found_fovea, contrast = find_landmarks(rgb)
    typed = disc is not None or fovea is not None
    disc, fovea = tuple(disc or found_disc), tuple(fovea or found_fovea)
    width_mm, nasal, matrix = placement(disc, fovea)
    corners = [reg.scan_to_photo(matrix, u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    if min(min(c) for c in corners) < 0 or max(max(c) for c in corners) > 1:
        raise SystemExit("The 6 mm square around the fovea does not fit inside the photo (landmarks wrong, or the photo "
                         "is not centred near the macula). Look at the photo and give --disc x y --fovea x y (0..1).")
    log(f"Disc at ({disc[0]:.2f}, {disc[1]:.2f}), fovea at ({fovea[0]:.2f}, {fovea[1]:.2f})"
        f"{' (typed in)' if typed else f', fovea {contrast:.0f} grey levels darker than around it'}; "
        f"disc on the {nasal}; photo width = {width_mm:.1f} mm IF disc-fovea = {DISC_FOVEA_MM} mm")
    if not typed and contrast < 4:
        log("WARNING: no clear dark spot where a fovea should be. Check landmarks.png; give --fovea x y if it is wrong.")

    if model:
        flat, meta = predict_thickness(model, macula_crop(rgb, matrix, nasal))
        thick = flat[:, ::-1] if nasal == "right" else flat               # back from 'disc on the left' to this photo
        method = "network"
        basis = (f"the average healthy map PLUS the correction of a network trained on {meta.get('pairs', '?')} pairs from "
                 f"'{meta.get('pairs_folder', '?')}' (its error on held-out pairs of that kind: {meta.get('test_mae_um', float('nan')):.0f} um). "
                 "It is only as good as those pairs are like this photo.")
        if meta.get("input") == "grey":
            basis += (" Those pairs had no colour (near-infrared pictures from the OCT machine), so this photo was turned "
                      "grey first; a colour photo turned grey is NOT an infrared picture, and the error above was "
                      "measured on infrared pictures only.")
    else:
        thick = atlas_thickness(OUT_PX, nasal)
        method = "atlas"
        basis = ("the thickness map of an AVERAGE HEALTHY eye, placed on this photo's fovea. No model looked at the "
                 "photo's content: the same map would come out for a diseased eye. It cannot show swelling or thinning.")

    # Points of interest in the photo: the disease model's heat map if one is given, else the classical spot filter.
    if heat_path:
        heat = np.asarray(Image.open(heat_path).convert("L").resize((512, 512), Image.Resampling.BILINEAR))
        spots, heat_source = None, f"the disease model's heat map {os.path.basename(heat_path)} (computed on the flat photo)"
    else:
        heat, spots = photo_spots(rgb, disc, fovea)
        heat_source = ("a classical filter for small spots that stand out (bright or dark; nothing learned, tuned on one "
                       "healthy photo and on cartoons): a hint where to look, not a finding")
        log(f"Points of interest (classical spot filter): {len(spots)}"
            + "".join(f"\n  {f['kind']} spot at ({f['x']:.2f}, {f['y']:.2f}), {f['z']:.0f}x the photo's usual variation" for f in spots[:8]))

    vessels = reg.warp_to_scan(reg.photo_vessel_map(rgb), matrix, 256)
    vessels = vessels / (np.percentile(vessels, 99.5) + 1e-6)
    vol, spacing, ilm, low = drawn_volume(thick, vessels)
    source = f"ESTIMATED from the photo {os.path.basename(photo_path)} ({method}): not a scan"
    check = landmarks_picture(rgb, disc, fovea, matrix)

    def fusion(thick_now):
        registration = {"matrix": matrix, "ncc": None, "control": None, "ratio": None, "trusted": True, "angle": 0.0,
                        "photo_mm_per_width": width_mm, "min_ncc": None, "min_ratio": None}
        page = {"photo_on_scan": oct3d.picture_url(np.clip(reg.warp_to_scan(rgb, matrix), 0, 255).astype(np.uint8), "JPEG", 88),
                "registration_check": oct3d.picture_url(check),
                "photo_full": oct3d.picture_url(np.asarray(Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR)), "JPEG", 88)}
        on_scan = reg.warp_to_scan(heat, matrix, 256) / 255.0             # the heat over the estimated square
        if heat.max() > 0:
            page["heat_on_scan"] = oct3d.picture_url((on_scan * 255).round().astype(np.uint8))
            page["heat_full"] = oct3d.picture_url(np.asarray(Image.fromarray(heat).resize((256, 256), Image.Resampling.BILINEAR)))
        # each point of interest with the ESTIMATED thickness under it
        hot = oct3d.heat_spots(on_scan, thick_now, spacing, vol.shape[0], vol.shape[2])
        return page, {"registration": registration, "heat_source": heat_source, "hot_spots": hot}

    page, summary, thick_full = oct3d.build_page_data(vol, spacing, ilm, low, source, fusion=fusion)
    # "Thickened" for an estimate = clearly thicker than the AVERAGE eye at the same place (more than 50 um above the
    # atlas), not "thicker than the rest of this scan": the atlas alone can therefore never produce one.
    atlas_full = np.asarray(Image.fromarray(atlas_thickness(OUT_PX, nasal), mode="F")
                            .resize(thick_full.shape[::-1], Image.Resampling.BILINEAR))
    findings, _ = oct3d.thick_findings(thick_full - atlas_full, spacing, 50.0)
    for f in findings:
        row, col = int(f["v"] * thick_full.shape[0]), int(f["u"] * thick_full.shape[1])
        f["excess_um"] = f["peak_um"]                                     # how far above the average eye, at its peak
        f["peak_um"] = float(thick_full[row, col])
    page["findings"] = summary["findings"] = findings
    info = {"method": method, "basis": basis, "points_of_interest": summary["fusion"]["hot_spots"],
            "points_source": heat_source, "spots_whole_photo": spots, "thickened_vs_average": findings, "photo": os.path.basename(photo_path),
            "disc": list(disc), "fovea": list(fovea), "landmarks_typed": typed, "fovea_contrast": contrast, "nasal": nasal,
            "assumed": {"disc_fovea_mm": DISC_FOVEA_MM, "photo_width_mm": width_mm, "field_mm": FIELD_MM,
                        "eye_wall": "drawn flat; a photo says nothing about its curve or tilt",
                        "layers": "painted at fixed fractions of the thickness; only the thickness is estimated"},
            "thickness_um": summary["thickness_um"], "model": os.path.abspath(model) if model else None}
    page["estimate"] = info                                               # the page shows its ESTIMATED banner from this

    os.makedirs(out, exist_ok=True)
    # The picture. The words "estimated ... not a scan" travel with it twice: as a thin line under it (unless
    # caption=False) and, always, inside the file's own text fields, because a bare grey picture looks like a real scan.
    from PIL.PngImagePlugin import PngInfo
    words = f"ESTIMATED from a fundus photo ({method}), not a scan"
    tags = PngInfo()
    tags.add_text("Description", words + ". " + basis)
    oct_picture(vol, words if caption else None).save(picture_path, pnginfo=tags)
    if want_page:
        with open(oct3d.TEMPLATE, encoding="utf-8") as f:
            html = f.read()
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(html.replace(oct3d.DATA_MARK, json.dumps(page).replace("</", "<\\/")))
        Image.fromarray(check).save(os.path.join(out, "landmarks.png"))
    np.save(os.path.join(out, "thickness_um.npy"), thick.astype(np.float32))
    with open(os.path.join(out, "estimate.json"), "w") as f:
        json.dump(info, f, indent=2)
    t = summary["thickness_um"]
    log(f"ESTIMATE ({method}): centre (1 mm) {t['central_1mm']:.0f} um, thinnest {t['min']:.0f}, thickest {t['max']:.0f}. Rests on: {basis}")
    for f in summary["fusion"]["hot_spots"]:
        log(f"  point of interest {f['id']}: {abs(f['x_mm']):.1f} mm {'right' if f['x_mm'] >= 0 else 'left'} of and "
            f"{abs(f['y_mm']):.1f} mm {'below' if f['y_mm'] >= 0 else 'above'} the fovea (as the photo is shown); "
            f"estimated retina under it {f['thickness_um']:.0f} um")
    for f in findings:
        log(f"  estimated thickening #{f['id']}: {f['excess_um']:.0f} um above the average eye, {f['area_mm2']:.1f} mm2")
    log(f"The estimated OCT picture: {picture_path}")
    if want_page:
        log(f"3D page: {page_path}  (it embeds the photo: do not commit it). Check {os.path.join(out, 'landmarks.png')} first.")
    return info


def make_pair(oct_out, photo_path, patient, nasal, pairs, name=None):
    """
    One REAL training pair from a finished `oct3d.py --scan ... --photo ...` folder: the photo
    as it lies over the scanned square (through the vessel alignment register.py found, which
    must be TRUSTED) + the thickness oct3d.py measured there. Only for scans of about 6 x 6 mm
    centred on the fovea: the network and the atlas assume that square.
    """
    with open(os.path.join(oct_out, "summary.json")) as f:
        summary = json.load(f)
    r = (summary.get("fusion") or {}).get("registration")
    if not r or not r["trusted"]:
        raise SystemExit(f"{oct_out}: no TRUSTED photo alignment (run oct3d.py with --photo; an untrusted match would teach nonsense).")
    w, l, _ = summary["size_mm"]
    if abs(w - FIELD_MM) > 0.6 or abs(l - FIELD_MM) > 0.6:
        raise SystemExit(f"{oct_out}: the scan covers {w:.1f} x {l:.1f} mm; pairs need about {FIELD_MM:.0f} x {FIELD_MM:.0f} mm.")
    rgb = np.asarray(Image.open(photo_path).convert("RGB"))
    thick = np.load(os.path.join(oct_out, "layers.npz"))["thickness_um"]
    thick = np.asarray(Image.fromarray(thick.astype(np.float32), mode="F").resize((OUT_PX, OUT_PX), Image.Resampling.BILINEAR))
    crop = macula_crop(rgb, r["matrix"], nasal)
    if nasal == "right":
        thick = thick[:, ::-1]
    add_pair(pairs, name or os.path.basename(os.path.normpath(oct_out)), patient, crop, thick)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="photo -> estimated thickness map and 3D page")
    e.add_argument("--photo", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--model", help="a model.pt from `train` (without it: the average healthy map)")
    e.add_argument("--disc", nargs=2, type=float, metavar=("X", "Y"), help="disc centre as fractions 0..1, if the automatic one is wrong")
    e.add_argument("--fovea", nargs=2, type=float, metavar=("X", "Y"))
    e.add_argument("--heat", help="the disease model's heat map for this photo (gray picture, same frame; eye3d.py writes "
                                  "them to <run>/eye3d/heat/). Without it: a classical spot filter marks points of interest")
    e.add_argument("--page", action="store_true", help="also write the 3D page (viewer.html) and landmarks.png")
    e.add_argument("--no-caption", action="store_true", help="leave out the thin 'estimated, not a scan' line under the picture")
    e.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("synthetic-pairs", help="cartoon pairs with a known rule, to test the code")
    s.add_argument("--out", required=True)
    s.add_argument("--eyes", type=int, default=300)
    m = sub.add_parser("make-pair", help="one real pair from an oct3d.py --photo output folder")
    m.add_argument("--oct", required=True)
    m.add_argument("--photo", required=True)
    m.add_argument("--patient", required=True, help="who this eye belongs to (both eyes of a person: the same name)")
    m.add_argument("--nasal", required=True, choices=["left", "right"], help="side of the PHOTO on which the optic disc lies")
    m.add_argument("--pairs", required=True)
    m.add_argument("--name")
    t = sub.add_parser("train", help="train ThicknessNet on a pairs folder")
    t.add_argument("--pairs", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--augment", action="store_true", help="flip upside down and shift each training pair at random")
    t.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    t.add_argument("--sector-weight", type=float, default=0.0, help="add this x the error of the 9 ETDRS sector averages to the loss")
    t.add_argument("--arch", choices=["unet", "resnet18"], default="unet", help="unet = ThicknessNet; resnet18 = model.py's ResNet as encoder")
    t.add_argument("--init", help="pretrain.py's encoder.pt (this project's own pretrained weights), with --arch resnet18")
    t.add_argument("--encoder-lr", type=float, default=0.1, help="the pretrained part's learning rate as a share of --lr (0 = frozen)")
    t.add_argument("--no-test", action="store_true", help="validation only, nothing saved: for choosing a recipe")
    args = p.parse_args()
    if args.cmd == "estimate":
        estimate(args.photo, args.out, args.model, args.disc, args.fovea, args.overwrite, heat_path=args.heat,
                 page=args.page, caption=not args.no_caption)
    elif args.cmd == "synthetic-pairs":
        for i in range(args.eyes):                                        # two eyes per cartoon patient
            crop, thick, _ = synthetic_pair(i)
            add_pair(args.out, f"p{i // 2:04d}_{'left' if i % 2 == 0 else 'right'}", f"p{i // 2:04d}", crop, thick)
        print(f"{args.eyes} cartoon pairs of {(args.eyes + 1) // 2} cartoon patients in {args.out}")
    elif args.cmd == "make-pair":
        make_pair(args.oct, args.photo, args.patient, args.nasal, args.pairs, args.name)
        print(f"Pair added to {args.pairs}")
    else:
        train(args.pairs, args.out, args.epochs, args.batch, args.lr, args.seed, augment=args.augment, wd=args.wd,
              test=not args.no_test, sector_weight=args.sector_weight, arch=args.arch, init=args.init,
              encoder_lr=args.encoder_lr)


if __name__ == "__main__":
    main()
