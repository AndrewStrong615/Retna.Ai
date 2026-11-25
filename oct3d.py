"""
True 3D from an OCT scan: the retina's MEASURED shape, in micrometres.

eye3d.py wraps one flat photo onto an assumed round eye: a drawing of depth.
This script reads an OCT volume instead. OCT (optical coherence tomography) sends
light into the retina and records how much comes back from each depth, a little
like ultrasound with light. One sweep gives a "B-scan": a cross-section picture,
depth going down. A stack of B-scans next to each other is a volume: real 3D data.

What this script does with a volume, all hand-written, nothing learned:

  1. READ it: a folder of B-scan pictures (how most public datasets ship), a .npy
     array [scan, depth, width], or a Heidelberg Spectralis .vol export.
  2. FIND TWO SURFACES in every B-scan:
       - the ILM, the retina's inner surface (dark vitreous above, bright tissue below);
       - the bright band at the bottom of the retina (photoreceptors / RPE).
     Both are "dark above, bright below" edges, the two strongest in a scan. Each
     is found as the best PATH from the left edge to the right edge: a path
     collects the edge strength of every pixel it passes and may move up or down
     only a few pixels per column, so it follows the real boundary and cannot jump
     to noise. The best such path is found exactly by dynamic programming (the
     idea of Chiu et al. 2010, written from scratch here).
  3. MEASURE: thickness = distance between the two surfaces, in micrometres, at
     every point: a thickness map. Thickened areas (as in swelling from diabetic
     macular edema) are listed as numbered findings with size and volume.
  4. WRITE ONE WEB PAGE (<out>/viewer.html, no internet needed) with the real
     surface as a 3D shape you can turn, shaded red where the retina is thick,
     a B-scan you can slide through the volume, and click-to-measure. Three shapes:
     "Scan only" (the default: nothing drawn; the photo's own colours and the OCT
     on a curved shell of radius 11 mm, height from the measured thickness inside
     the scanned square only, the OCT's real cross-sections on the square's cut
     edges, a 1 mm grid; the curve is an ASSUMPTION, and relief guessed from the
     photo's brightness is a separate switch, off by default, labelled estimated),
     the scan curved onto the back wall of a textbook-style cut-open standard eye
     (wall coats, cornea, iris, lens, ciliary body, muscles and optic nerve are a
     DRAWING with named parts, the same for every scan; the thickness on it is
     measured, with a 1x-5x relief slider), or the flat block exactly as scanned.

WHAT IS REAL AND WHAT IS NOT:
  - The shape and the micrometres are measured from the scan (given the right
    pixel spacing: read from a .vol file, otherwise the --width-mm / --length-mm /
    --depth-mm you pass; the defaults are a typical 6 x 6 x 1.9 mm scan and are
    marked "assumed" in the page).
  - The surface finder is a classical method and has ONLY been checked on a
    synthetic volume with a known answer, not yet on real scans. On real scans it
    can lose the surface in shadows, in very swollen or detached retina, and at
    the optic disc. A .vol file carries the device's own surfaces: when present,
    the script reports how far ours are from them (a free check, no labels needed).
  - Our lower surface is the TOP of the bright band; devices usually measure to
    its bottom (Bruch's membrane), so their thickness numbers run some tens of
    micrometres higher.
  - "Red" means thicker than the rest of THIS scan (or than --thick-um). There is
    no database of normal eyes behind it. It is a measurement, not a diagnosis.
  - The disease model (train.py) reads fundus PHOTOS. It has never seen OCT.
    With --photo (and --heat) the photo of the SAME eye is lined up with the
    scan by its vessels (register.py) and laid onto the measured surface, and
    the model's heat map rides along. The heat is still computed on the flat
    photo: the model does not look at the depth. If the vessel match is weak,
    the script refuses to fuse (--force-fusion overrides, and the page says so).

Privacy: a .vol file contains the patient's ID and birth date. This script does
not read them out, and the page contains only pictures and measurements. Still,
treat scans and pages like the photos: never commit them to git.

Usage:
  python oct3d.py --demo --out runs\\oct_demo                 (synthetic volume, to try the page)
  python oct3d.py --scan <folder of B-scans> --out runs\\oct_eye1 --width-mm 6 --length-mm 6 --depth-mm 1.9
  python oct3d.py --scan eye1.vol --out runs\\oct_eye1
  python oct3d.py --scan <scan> --out runs\\oct_eye1 --photo <processed photo>.jpg --heat runs\\exp15\\eye3d\\heat\\<name>.png
    (photo + heat of the same eye: eye3d.py writes the heat pictures; both must be the SAME preprocessed photo frame)
Writes viewer.html, layers.npz (surfaces and thickness as arrays) and summary.json.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import struct

import numpy as np
from PIL import Image, ImageFilter

from eye3d import label_regions
import register as reg

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "oct3d_viewer.html")
DATA_MARK = "/*OCT_DATA*/"
PICTURES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# 1. Reading volumes.  A volume is a uint8 array [scan, depth, width]:
#    scan = which B-scan (the slow direction), depth 0 = top (vitreous side).
# ---------------------------------------------------------------------------

def read_folder(folder):
    """A folder of B-scan pictures, in the order of the numbers in their names (2 before 10)."""
    names = [n for n in os.listdir(folder) if n.lower().endswith(PICTURES)]
    if len(names) < 2:
        raise SystemExit(f"{folder}: need at least 2 B-scan pictures, found {len(names)}.")
    names.sort(key=lambda n: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", n)])
    scans = [np.asarray(Image.open(os.path.join(folder, n)).convert("L")) for n in names]
    shapes = {s.shape for s in scans}
    if len(shapes) != 1:
        raise SystemExit(f"{folder}: the pictures have different sizes {sorted(shapes)}; "
                         "one volume = B-scans of one size (is this several scans mixed together?).")
    return np.stack(scans)


def read_vol(path):
    """
    Heidelberg Spectralis raw export (.vol). Layout, all little-endian:
      file header, 2048 bytes: sizes, pixel spacing in mm, and patient details (not read);
      the SLO picture (the device's own fundus view), SizeXSlo * SizeYSlo bytes;
      then per B-scan: a header (BScanHdrSize bytes), inside it at OffSeg the device's own
      surfaces (NumSeg lines of SizeX float32: line 0 = ILM, line 1 = Bruch's membrane),
      then SizeZ * SizeX float32 reflectivities in 0..1 (3.4e38 = no data).
    Returns (volume, spacing in micrometres, device surfaces or None).
    NOT yet tested on a real file: written from the published format description.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if not raw[:12].startswith(b"HSF-OCT"):
        raise SystemExit(f"{path} does not start with 'HSF-OCT': not a Heidelberg .vol export.")
    size_x, n_scans, size_z, scale_x, distance, scale_z, slo_x, slo_y = struct.unpack_from("<iiidddii", raw, 12)
    # after SizeYSlo: ScaleXSlo d, ScaleYSlo d, FieldSizeSlo i, ScanFocus d, ScanPosition 4s, ExamTime q, ScanPattern i
    hdr_size = struct.unpack_from("<i", raw, 12 + 44 + 16 + 4 + 8 + 4 + 8 + 4)[0]
    if not (0 < size_x < 10000 and 0 < n_scans < 2000 and 0 < size_z < 10000 and 0 < hdr_size < 100000):
        raise SystemExit(f"{path}: implausible header (width {size_x}, scans {n_scans}, depth {size_z}).")
    start = 2048 + slo_x * slo_y
    block = hdr_size + size_x * size_z * 4
    if len(raw) < start + n_scans * block:
        raise SystemExit(f"{path}: file is shorter than its header says ({len(raw)} bytes).")
    vol = np.zeros((n_scans, size_z, size_x), dtype=np.uint8)
    device = np.full((2, n_scans, size_x), np.nan, dtype=np.float32)
    for b in range(n_scans):
        at = start + b * block
        n_seg, off_seg = struct.unpack_from("<ii", raw, at + 12 + 4 + 32)
        if n_seg >= 2 and 0 < off_seg and off_seg + 2 * size_x * 4 <= hdr_size:
            seg = np.frombuffer(raw, dtype="<f4", count=2 * size_x, offset=at + off_seg).reshape(2, size_x)
            device[:, b] = np.where(seg < 1e30, seg, np.nan)
        data = np.frombuffer(raw, dtype="<f4", count=size_x * size_z, offset=at + hdr_size).reshape(size_z, size_x)
        # Reflectivity spans several orders of magnitude; the 4th root is the usual display curve.
        vol[b] = (np.clip(np.where(data < 1e30, data, 0), 0, 1) ** 0.25 * 255).astype(np.uint8)
    spacing = {"dx": scale_x * 1000, "dy": distance * 1000, "dz": scale_z * 1000, "assumed": False}
    return vol, spacing, (device if np.isfinite(device).any() else None)


def load_volume(path, width_mm=6.0, length_mm=6.0, depth_mm=1.9):
    """Returns (volume uint8 [scan, depth, width], spacing dict in micrometres, device surfaces or None)."""
    device = None
    if os.path.isdir(path):
        vol, spacing = read_folder(path), None
    elif path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 3:
            raise SystemExit(f"{path}: expected a 3-D array [scan, depth, width], got shape {arr.shape}.")
        arr = arr.astype(np.float32)
        vol = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        spacing = None
    elif path.lower().endswith(".vol"):
        vol, spacing, device = read_vol(path)
    else:
        raise SystemExit(f"{path}: give a folder of B-scan pictures, a .npy array or a Heidelberg .vol file. "
                         "(.e2e and DICOM are not read yet.)")
    if spacing is None:                           # no spacing in the data: use the sizes given on the command line
        n, z, x = vol.shape
        spacing = {"dx": width_mm * 1000 / x, "dy": length_mm * 1000 / n, "dz": depth_mm * 1000 / z, "assumed": True}
    return vol, spacing, device


def synthetic_volume(n_scans=48, depth=256, width=256, seed=0, vessel_mask=None):
    """
    A cartoon OCT volume with a KNOWN answer, to test the surface finder and to try the page.
    6 x 6 x 2 mm. The retina is a bright slab, 300 um thick, on a gently curved, tilted eye wall;
    a dip in the middle (the fovea) and one swollen area with a dark fluid pocket inside,
    1.5 mm right of and 1 mm behind the centre, 200 um high. Speckle noise on top, as in real OCT.
    `vessel_mask` ([scan, width], 0..1): where vessels run; everything below them is darkened
    (their shadows), as in real OCT. None = no vessels.
    Returns (volume, spacing, true ILM, true lower surface), surfaces in pixels [scan, width].
    """
    rng = np.random.default_rng(seed)
    spacing = {"dx": 6000 / width, "dy": 6000 / n_scans, "dz": 2000 / depth, "assumed": False}
    x = (np.arange(width) + 0.5) * spacing["dx"] / 1000 - 3            # mm from the centre
    y = (np.arange(n_scans) + 0.5) * spacing["dy"] / 1000 - 3
    yy, xx = np.meshgrid(y, x, indexing="ij")
    wall = 900 - 12 * (xx ** 2 + yy ** 2) + 25 * xx                    # depth of the lower surface, um
    pit = 110 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.35 ** 2))         # fovea: the inner surface dips
    swell = 200 * np.exp(-((xx - 1.5) ** 2 + (yy + 1.0) ** 2) / (2 * 0.5 ** 2))
    ilm_um, low_um = wall - 300 + pit - swell, wall
    z = (np.arange(depth) + 0.5) * spacing["dz"]                       # um, shape [depth]
    zz = z[None, :, None]
    ilm, low = ilm_um[:, None, :], low_um[:, None, :]
    img = np.full((n_scans, depth, width), 14.0)                       # vitreous: nearly black
    img = np.where(zz >= ilm, 105.0, img)                              # retina
    img = np.where((zz >= ilm) & (zz < ilm + 35), 175.0, img)          # nerve fibre layer: bright
    img = np.where((zz >= ilm + 110) & (zz < ilm + 150), 70.0, img)    # a darker nuclear layer
    img = np.where((zz >= low - 60) & (zz < low), 60.0, img)           # outer nuclear layer: dark, above the band
    img = np.where((zz >= low) & (zz < low + 45), 225.0, img)          # photoreceptor / RPE band: brightest
    img = np.where(zz >= low + 45, 120.0 * np.exp(-(zz - low - 45) / 400), img)   # choroid, fading with depth
    pocket = (((xx - 1.5) / 0.35) ** 2 + ((yy + 1.0) / 0.35) ** 2)[:, None, :] + ((zz - (low - 190)) / 70) ** 2 < 1
    img = np.where(pocket & (zz >= ilm + 35), 22.0, img)               # fluid: dark
    if vessel_mask is not None:
        img = np.where(zz >= ilm + 40, img * (1 - 0.6 * vessel_mask[:, None, :]), img)
    img = img * rng.exponential(1.0, img.shape) ** 0.5 * 1.13 + rng.normal(0, 6, img.shape)   # speckle
    return (np.clip(img, 0, 255).astype(np.uint8), spacing,
            (ilm_um / spacing["dz"] - 0.5).astype(np.float32), (low_um / spacing["dz"] - 0.5).astype(np.float32))


def synthetic_vessels(seed=0, tree=0):
    """Cartoon vessel paths in millimetres (0, 0 = centre of the scan; x right, y down):
    wandering, branching lines that start at an 'optic disc' 4 mm left of the centre.
    Returns a list of (points [n, 2], width in mm). `tree` picks a different pattern."""
    rng = np.random.default_rng(1000 + 17 * seed + tree)
    disc = np.array([-4.0, 0.4])
    todo = [(disc.copy(), a, 0.13) for a in np.linspace(0, 2 * np.pi, 9)[:-1] + rng.uniform(-0.3, 0.3, 8)]
    paths = []
    while todo and len(paths) < 60:
        pos, heading, width = todo.pop()
        points = [pos.copy()]
        for step in range(int(rng.integers(25, 70))):
            heading += rng.normal(0, 0.16)
            pos = pos + 0.15 * np.array([np.cos(heading), np.sin(heading)])
            points.append(pos.copy())
            if width > 0.06 and rng.random() < 0.05:                # a branch, a little thinner
                todo.append((pos.copy(), heading + rng.choice([-1, 1]) * rng.uniform(0.5, 1.0), width * 0.75))
            if np.hypot(*pos) > 8:
                break
        paths.append((np.array(points), width))
    return paths


def draw_paths(paths, to_pixel, size, scale_px_per_mm):
    """Rasterize vessel paths into a size x size picture (0..1); to_pixel maps mm points -> pixels."""
    from PIL import ImageDraw
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for points, width in paths:
        d.line([tuple(q) for q in to_pixel(points)], fill=255, width=max(1, round(width * scale_px_per_mm)))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255


def synthetic_pair(seed=0, tree_photo=0, tree_scan=0):
    """
    A cartoon PHOTO + SCAN of the same cartoon eye, with a known alignment, to test register.py.
    The scan is synthetic_volume() with vessel shadows. The photo covers a 12 mm circle, is
    centred 0.7 mm right / 0.4 mm above the scan's centre and turned by 6 degrees; it shows
    the same vessels (when tree_photo == tree_scan), a bright disc, and a few spots at the
    swollen area. Also returns a DRAWN heat map (a blob at those spots: a stand-in, NOT from
    a model) and the true matrix (scan position 0..1 -> photo position 0..1).
    """
    n_scans, width, hi = 48, 256, 512
    scan_paths = synthetic_vessels(seed, tree_scan)
    mask_hi = draw_paths(scan_paths, lambda p: (p + 3) / 6 * hi, hi, hi / 6)          # the 6 x 6 mm scan square
    rows = np.clip(((np.arange(n_scans) + 0.5) / n_scans * hi).astype(int), 0, hi - 1)
    mask = mask_hi[rows].reshape(n_scans, width, hi // width).mean(axis=2)
    vol, spacing, ilm, low = synthetic_volume(n_scans=n_scans, width=width, seed=seed, vessel_mask=mask)

    size, big = 384, 768
    mm_per_px = 12 / (0.94 * big)
    centre_mm, turn = np.array([0.7, -0.4]), np.radians(6.0)
    cos, sin = np.cos(turn), np.sin(turn)
    rot = np.array([[cos, sin], [-sin, cos]])                         # counter-clockwise on a screen (y down)
    to_photo = lambda p: (np.atleast_2d(p) - centre_mm) / mm_per_px @ rot.T + big / 2
    rng = np.random.default_rng(500 + seed)
    yy, xx = np.mgrid[:big, :big]
    r = np.hypot(xx - big / 2, yy - big / 2) / (0.47 * big)
    photo = np.stack([205 - 45 * r ** 2, 112 - 35 * r ** 2, 62 - 20 * r ** 2], axis=-1)
    vessels = draw_paths(synthetic_vessels(seed, tree_photo), to_photo, big, 1 / mm_per_px)[..., None]
    photo = photo * (1 - vessels) + np.array([105.0, 28.0, 24.0]) * vessels
    dx, dy = to_photo(np.array([-4.0, 0.4]))[0]
    disc = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * (0.75 / mm_per_px) ** 2))[..., None]
    photo = photo * (1 - 0.8 * disc) + np.array([250.0, 225.0, 170.0]) * 0.8 * disc
    lesion = to_photo(np.array([1.5, -1.0]))[0]
    for _ in range(9):                                                # dark and yellow spots at the swollen area
        sx, sy = lesion + rng.normal(0, 0.35 / mm_per_px, 2)
        spot = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.07 / mm_per_px) ** 2))[..., None]
        photo = photo * (1 - spot) + np.array([[120.0, 20.0, 20.0], [240.0, 215.0, 110.0]][int(rng.random() < 0.4)]) * spot
    photo = np.where((r <= 1)[..., None], photo + rng.normal(0, 4, photo.shape), 0)
    photo = np.asarray(Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS))
    heat = np.exp(-((xx - lesion[0]) ** 2 + (yy - lesion[1]) ** 2) / (2 * (0.5 / mm_per_px) ** 2))
    heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR))
    # True matrix: scan (u, v) -> mm (6u - 3, 6v - 3) -> photo pixels -> 0..1
    k = 6 / mm_per_px / big
    shift = (rot @ ((np.array([-3.0, -3.0]) - centre_mm) / mm_per_px) + big / 2) / big
    matrix = np.array([[cos * k, sin * k, shift[0]], [-sin * k, cos * k, shift[1]]])
    return vol, spacing, ilm, low, photo, heat, matrix


# ---------------------------------------------------------------------------
# 2. Finding the surfaces
# ---------------------------------------------------------------------------

def best_path(score, jump=3, bend_cost=0.5):
    """
    For every B-scan at once: the path from the left edge to the right edge, one depth per
    column, with the largest total `score`, moving at most `jump` pixels up or down per
    column (each pixel of movement costs `bend_cost`, so straight is preferred when in doubt).
    score: [scan, depth, width]. Returns depths [scan, width] (int).

    Dynamic programming: best[z] = the best total of any path ending at depth z in the
    current column. Moving one column right, a path can arrive at z from z - jump ... z + jump,
    so  new_best[z] = score[z] + max over those of (best[from] - cost).  Remembering which
    "from" won at every step lets us walk the winning path back from right to left.
    This finds the true optimum; trying every path one by one would be impossible.
    """
    n, depth, width = score.shape
    shifts = np.arange(-jump, jump + 1)
    best = score[:, :, 0].astype(np.float32)
    came_from = np.zeros((width, n, depth), dtype=np.int8)
    for x in range(1, width):
        padded = np.pad(best, ((0, 0), (jump, jump)), constant_values=-np.inf)
        # options[k][z] = best[z + shifts[k]] minus the cost of that move
        options = np.stack([padded[:, jump + s: jump + s + depth] - bend_cost * abs(s) for s in shifts])
        pick = options.argmax(axis=0)
        came_from[x] = shifts[pick]
        best = np.take_along_axis(options, pick[None], axis=0)[0] + score[:, :, x]
    path = np.zeros((n, width), dtype=np.int64)
    z = best.argmax(axis=1)
    rows = np.arange(n)
    for x in range(width - 1, -1, -1):
        path[:, x] = z
        z = z + came_from[x][rows, z]
    return path


def median3x3(surface):
    """Each value replaced by the median of itself and its 8 neighbours: removes single-scan outliers."""
    p = np.pad(surface, 1, mode="edge")
    h, w = surface.shape
    return np.median(np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)]), axis=0)


def find_layers(vol, spacing, jump=3):
    """
    The two surfaces of every B-scan, in pixels [scan, width]: (ILM, lower surface).
    Steps: blur away the speckle; edge strength = brightness a little BELOW minus a little
    ABOVE each pixel (large where dark turns to bright going down); best path = one of the
    two surfaces; forbid a band around it and take the best path again = the other one;
    the upper of the two is the ILM. Finally a 3x3 median across neighbouring scans.
    """
    smooth = np.stack([np.asarray(Image.fromarray(s).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
                       for s in vol])
    k = max(2, round(12 / spacing["dz"]))                 # compare ~12 um below with ~12 um above
    edge = np.zeros_like(smooth)
    edge[:, k:-k] = smooth[:, 2 * k:] - smooth[:, :-2 * k]
    first = best_path(edge, jump)
    band = max(jump + 2, round(60 / spacing["dz"]))       # the two surfaces are > 60 um apart
    depth_index = np.arange(vol.shape[1])[None, :, None]
    blocked = np.where(np.abs(depth_index - first[:, None, :]) <= band, -1e6, edge)
    second = best_path(blocked, jump)
    first_is_upper = (first.mean(axis=1) < second.mean(axis=1))[:, None]
    ilm = np.where(first_is_upper, first, second).astype(np.float32)
    low = np.where(first_is_upper, second, first).astype(np.float32)
    return median3x3(ilm), median3x3(low)


def thick_findings(thick_um, spacing, threshold_um, max_findings=8, min_area_mm2=0.05):
    """
    Numbered thickened areas: connected regions of the thickness map above `threshold_um`,
    largest extra volume first. Returns (findings, label picture). For each: where its centre
    is (mm from the scan's centre, and which B-scan), its area, its thickest point, and the
    EXTRA volume: the tissue above the threshold, added up (um x mm2 -> mm3).
    """
    cell_mm2 = spacing["dx"] * spacing["dy"] / 1e6
    labels, count = label_regions(thick_um > threshold_um)
    n, width = thick_um.shape
    found = []
    for k in range(1, count + 1):
        region = labels == k
        area = float(region.sum() * cell_mm2)
        if area < min_area_mm2:
            continue
        extra = thick_um[region] - threshold_um
        bs, xs = np.nonzero(region)
        cb, cx = float((bs * extra).sum() / extra.sum()), float((xs * extra).sum() / extra.sum())
        found.append({
            "scan": int(round(cb)), "u": (cx + 0.5) / width, "v": (cb + 0.5) / n,
            "x_mm": (cx + 0.5 - width / 2) * spacing["dx"] / 1000, "y_mm": (cb + 0.5 - n / 2) * spacing["dy"] / 1000,
            "area_mm2": area, "peak_um": float(thick_um[region].max()),
            "extra_volume_mm3": float(extra.sum() * cell_mm2 / 1000), "_label": k,
        })
    found.sort(key=lambda f: -f["extra_volume_mm3"])
    found = found[:max_findings]
    numbered = np.zeros(labels.shape, dtype=np.uint8)
    for number, f in enumerate(found, start=1):
        numbered[labels == f.pop("_label")] = number
        f["id"] = number
    return found, numbered


def default_threshold(thick_um):
    """'Clearly thicker than the rest of this scan': the median plus 2.5 robust standard
    deviations (1.4826 x the median distance from the median), and at least 50 um above it."""
    med = float(np.median(thick_um))
    spread = 1.4826 * float(np.median(np.abs(thick_um - med)))
    return med + max(2.5 * spread, 50.0)


# ---------------------------------------------------------------------------
# 3. Writing the page
# ---------------------------------------------------------------------------

def picture_url(array, fmt="PNG", quality=82):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {"optimize": True}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def floats_b64(array):
    """A float array as text (little-endian float32, base64), which the page turns back into numbers."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def resample_rows_cols(surface, rows, cols):
    """Pick evenly spaced scans and average neighbouring columns, to at most rows x cols points."""
    n, width = surface.shape
    surface = surface[np.round(np.linspace(0, n - 1, rows)).astype(int)]
    edges = np.linspace(0, width, cols + 1).astype(int)
    return np.stack([surface[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def fuse(vol, low, thick_um, spacing, photo_rgb, heat=None, photo_width_mm=13.0, heat_source="", hot=0.5):
    """
    Line the photo up with the scan (register.py) and carry the photo, and its heat map if
    given, onto the scan's square. Returns (page entries, summary entries).
    Model hot spots = connected regions of the carried-over heat at or above `hot` of the
    PHOTO's strongest spot; for each, the measured retina thickness under it: the one thing
    neither the photo nor the scan could say alone.
    """
    n, _, width = vol.shape
    shadow = reg.oct_shadow_picture(vol, low, spacing)
    found = reg.register(photo_rgb, reg.oct_vessel_map(shadow), width * spacing["dx"] / 1000, photo_width_mm)
    summary = {"registration": {k: found[k] for k in ("matrix", "ncc", "control", "ratio", "trusted", "angle",
                                                      "photo_mm_per_width", "min_ncc", "min_ratio")},
               "heat_source": heat_source if heat is not None else None, "hot_spots": []}
    page = {"photo_on_scan": picture_url(reg.warp_to_scan(photo_rgb, found["matrix"]).astype(np.uint8), "JPEG", 88),
            "registration_check": picture_url(reg.check_picture(photo_rgb, shadow, found["matrix"])),
            # the whole photo (and heat), for the eyeball view's back wall
            "photo_full": picture_url(np.asarray(Image.fromarray(photo_rgb).resize((512, 512), Image.Resampling.BILINEAR)), "JPEG", 88)}
    if heat is not None:
        on_scan = reg.warp_to_scan(heat, found["matrix"], 256) / 255.0           # 0..1 of the photo's strongest spot
        page["heat_on_scan"] = picture_url((on_scan * 255).round().astype(np.uint8))
        page["heat_full"] = picture_url(np.asarray(Image.fromarray(heat).resize((256, 256), Image.Resampling.BILINEAR)))
        thick_256 = np.asarray(Image.fromarray(thick_um.astype(np.float32), mode="F")
                               .resize((256, 256), Image.Resampling.BILINEAR))
        labels, count = label_regions(on_scan >= hot)
        cell_mm2 = (width * spacing["dx"] / 1000 / 256) * (n * spacing["dy"] / 1000 / 256)
        spots = []
        for k in range(1, count + 1):
            region = labels == k
            if region.sum() * cell_mm2 < 0.02:
                continue
            w = on_scan[region]
            vs, us = np.nonzero(region)
            u, v = float(((us + 0.5) * w).sum() / w.sum() / 256), float(((vs + 0.5) * w).sum() / w.sum() / 256)
            spots.append({"u": u, "v": v, "scan": int(min(n - 1, v * n)),
                          "x_mm": (u - 0.5) * width * spacing["dx"] / 1000, "y_mm": (v - 0.5) * n * spacing["dy"] / 1000,
                          "area_mm2": float(region.sum() * cell_mm2), "peak_heat": float(w.max()),
                          "thickness_um": float((thick_256[region] * w).sum() / w.sum()),
                          "thickness_max_um": float(thick_256[region].max()), "_mass": float(w.sum())})
        spots.sort(key=lambda f: -f.pop("_mass"))
        for letter, f in zip("ABCDEFGH", spots):
            f["id"] = letter
        summary["hot_spots"] = spots[:8]
        summary["heat_outside_scan_pct"] = 100 * float(1 - on_scan.sum() * (heat.size / 256 ** 2) *
                                                       abs(np.linalg.det(np.array(found["matrix"])[:, :2]))
                                                       / (heat.sum() / 255.0 + 1e-8))
    return page, summary


def build_page_data(vol, spacing, ilm, low, source, threshold_um=None, max_slices=64, grid=160, device_check=None,
                    fusion=None):
    n, depth, width = vol.shape
    thick = (low - ilm) * spacing["dz"]
    threshold = default_threshold(thick) if threshold_um is None else float(threshold_um)
    findings, _ = thick_findings(thick, spacing, threshold)

    # Centre of the scan: mean thickness within 0.5 mm (the "central 1 mm circle" eye doctors
    # quote, IF the scan is centred on the fovea, which this script cannot know).
    y_mm = (np.arange(n) + 0.5 - n / 2) * spacing["dy"] / 1000
    x_mm = (np.arange(width) + 0.5 - width / 2) * spacing["dx"] / 1000
    central = np.hypot(*np.meshgrid(y_mm, x_mm, indexing="ij")) <= 0.5

    rows, cols = min(n, grid), min(width, grid)
    en_face = vol.mean(axis=1)                                           # looking down through the volume
    en_face = ((en_face - en_face.min()) / (en_face.max() - en_face.min() + 1e-8) * 255).astype(np.uint8)
    keep = np.unique(np.round(np.linspace(0, n - 1, min(n, max_slices))).astype(int))
    summary = {
        "source": source, "scans": n, "depth_px": depth, "width_px": width,
        "spacing_um": {k: float(spacing[k]) for k in ("dx", "dy", "dz")}, "spacing_assumed": bool(spacing["assumed"]),
        "size_mm": [width * spacing["dx"] / 1000, n * spacing["dy"] / 1000, depth * spacing["dz"] / 1000],
        "thickness_um": {"mean": float(thick.mean()), "median": float(np.median(thick)), "max": float(thick.max()),
                         "min": float(thick.min()),
                         "central_1mm": float(thick[central].mean()) if central.any() else None},
        "threshold_um": threshold, "threshold_given": threshold_um is not None,
        "findings": findings, "device_check": device_check,
    }
    page = dict(summary)
    page.update({
        "grid": [rows, cols],
        "ilm_um": floats_b64(resample_rows_cols(ilm, rows, cols) * spacing["dz"]),      # depth below the top of the scan
        "low_um": floats_b64(resample_rows_cols(low, rows, cols) * spacing["dz"]),
        "en_face": picture_url(en_face),
        "slices": [{"scan": int(b), "img": picture_url(vol[b], "JPEG")} for b in keep],
    })
    if fusion is not None:
        fusion_page, fusion_summary = fusion(thick)
        summary["fusion"] = fusion_summary
        page["fusion"] = {**fusion_summary, **fusion_page}
    return page, summary, thick


def compare_with_device(ilm, low, device, spacing):
    """Mean distance (um) between our surfaces and the device's own (.vol files only)."""
    out = {}
    for name, ours, theirs in (("ilm", ilm, device[0]), ("lower_vs_bruch", low, device[1])):
        ok = np.isfinite(theirs)
        if ok.any():
            diff = (ours[ok] - theirs[ok]) * spacing["dz"]
            out[name] = {"mean_abs_um": float(np.abs(diff).mean()), "mean_um": float(diff.mean())}
    return out or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", help="a folder of B-scan pictures, a .npy array [scan, depth, width], or a Heidelberg .vol")
    p.add_argument("--demo", action="store_true", help="use the built-in synthetic volume instead of --scan")
    p.add_argument("--out", required=True, help="folder for viewer.html, layers.npz and summary.json")
    p.add_argument("--width-mm", type=float, default=6.0, help="width of one B-scan (ignored for .vol)")
    p.add_argument("--length-mm", type=float, default=6.0, help="distance from the first to the last B-scan")
    p.add_argument("--depth-mm", type=float, default=1.9, help="depth covered by one B-scan, top to bottom")
    p.add_argument("--thick-um", type=float, default=None,
                   help="show red / list findings above this thickness (default: clearly thicker than the rest of the scan)")
    p.add_argument("--max-slices", type=int, default=64, help="how many B-scans to put into the page")
    p.add_argument("--jump", type=int, default=3, help="how many pixels a surface may move per column")
    p.add_argument("--photo", help="fundus photo of the SAME eye: lined up with the scan by its vessels and laid on the surface")
    p.add_argument("--heat", help="the model's heat map for that photo (gray picture, same frame as --photo; "
                                  "eye3d.py writes them to <run>/eye3d/heat/)")
    p.add_argument("--photo-width-mm", type=float, default=13.0,
                   help="first guess of the retina width the photo covers (the search tries 0.6x to 1.6x of it)")
    p.add_argument("--force-fusion", action="store_true",
                   help="fuse even when the vessel match is not trusted (the page then carries a warning)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if bool(args.scan) == args.demo:
        raise SystemExit("Give either --scan <path> or --demo.")
    page_path = os.path.join(args.out, "viewer.html")
    if os.path.exists(page_path) and not args.overwrite:
        raise SystemExit(f"{page_path} exists. Pick a new --out or add --overwrite.")

    if args.heat and not args.photo:
        raise SystemExit("--heat needs --photo (the heat map belongs to a photo).")
    photo = heat = None
    heat_source = ""
    if args.demo:
        vol, spacing, true_ilm, true_low, photo, heat, _ = synthetic_pair()
        device, source = None, "synthetic demo: a cartoon scan + cartoon photo, not an eye"
        heat_source = "DRAWN for the demo (a blob at the cartoon spots), not from a model"
    else:
        vol, spacing, device = load_volume(args.scan, args.width_mm, args.length_mm, args.depth_mm)
        source = os.path.basename(os.path.normpath(args.scan))
        if args.photo:
            photo = np.asarray(Image.open(args.photo).convert("RGB"))
            if photo.shape[0] != photo.shape[1]:
                raise SystemExit(f"{args.photo}: expected a square (preprocessed) photo, got {photo.shape[1]} x {photo.shape[0]}.")
            if args.heat:
                heat = np.asarray(Image.open(args.heat).convert("L").resize(photo.shape[:2][::-1]))
                heat_source = f"{os.path.basename(args.heat)} (computed on the flat photo)"
    n, depth, width = vol.shape
    how = "ASSUMED from --width-mm/--length-mm/--depth-mm" if spacing["assumed"] else "from the data"
    print(f"Volume: {n} B-scans of {width} x {depth} px; spacing {spacing['dx']:.1f} x {spacing['dy']:.1f} x "
          f"{spacing['dz']:.2f} um ({how})")

    ilm, low = find_layers(vol, spacing, args.jump)
    check = None
    if args.demo:
        print(f"Against the known answer: ILM off by {np.abs(ilm - true_ilm).mean() * spacing['dz']:.1f} um on average, "
              f"lower surface by {np.abs(low - true_low).mean() * spacing['dz']:.1f} um")
    if device is not None:
        check = compare_with_device(ilm, low, device, spacing)
        if check:
            print("Against the device's own surfaces (mean distance): " +
                  ", ".join(f"{k} {v['mean_abs_um']:.1f} um" for k, v in check.items()))

    fusion = None
    if photo is not None:
        fusion = lambda thick: fuse(vol, low, thick, spacing, photo, heat, args.photo_width_mm, heat_source)
    page, summary, thick = build_page_data(vol, spacing, ilm, low, source, args.thick_um, args.max_slices,
                                           device_check=check, fusion=fusion)
    if fusion is not None:
        r = summary["fusion"]["registration"]
        print(f"Photo lined up with the scan: fit {r['ncc']:.2f} (control {r['control']:.2f}, ratio {r['ratio']:.1f}), "
              f"turned {r['angle']:+.0f} degrees, photo width = {r['photo_mm_per_width']:.1f} mm -> "
              + ("TRUSTED" if r["trusted"] else "NOT TRUSTED"))
        if not r["trusted"] and not args.force_fusion:
            raise SystemExit("The vessel patterns of the photo and the scan do not clearly match (same eye? right "
                             "--photo-width-mm? mirrored?). Nothing written. --force-fusion writes the page anyway, "
                             "with a warning.")
        for f in summary["fusion"]["hot_spots"]:
            print(f"  model hot spot {f['id']}: heat {f['peak_heat'] * 100:.0f}%, retina under it {f['thickness_um']:.0f} um "
                  f"(scan median {summary['thickness_um']['median']:.0f}), at x {f['x_mm']:+.1f} mm, scan {f['scan'] + 1}")
    t = summary["thickness_um"]
    print(f"Thickness: median {t['median']:.0f} um, centre {t['central_1mm']:.0f} um, max {t['max']:.0f} um; "
          f"red above {summary['threshold_um']:.0f} um; {len(summary['findings'])} thickened area(s)")
    for f in summary["findings"]:
        print(f"  #{f['id']}: {f['area_mm2']:.2f} mm2, up to {f['peak_um']:.0f} um, extra volume "
              f"{f['extra_volume_mm3']:.3f} mm3, at x {f['x_mm']:+.1f} mm, scan {f['scan'] + 1}")

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if DATA_MARK not in html:
        raise SystemExit(f"{TEMPLATE} has no {DATA_MARK} marker.")
    os.makedirs(args.out, exist_ok=True)
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(html.replace(DATA_MARK, json.dumps(page).replace("</", "<\\/")))
    np.savez_compressed(os.path.join(args.out, "layers.npz"), ilm_px=ilm, lower_px=low, thickness_um=thick,
                        spacing_um=np.array([spacing["dx"], spacing["dy"], spacing["dz"]]))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Open in a browser: {page_path}  ({os.path.getsize(page_path) / 1e6:.1f} MB; do not commit it)")


if __name__ == "__main__":
    main()
