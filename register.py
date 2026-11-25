"""
Registration: lining a fundus photo up with an OCT volume, using the vessels.

A fundus photo shows the retina from the front: a flat picture with detail and
colour, where our disease model finds its evidence. An OCT volume holds the real
depth, but only of a smaller square (often 6 x 6 mm around the centre). To put
the photo, and the model's heat map, onto the measured 3D surface, we must know
which pixel of the photo belongs to which point of the scan. Blood vessels are
the landmarks both share:

  - in the PHOTO, vessels are thin dark lines (darkest in the green channel);
  - in the OCT, vessels sit in the inner retina and absorb the light, so they
    cast SHADOWS on everything below them. Averaging the brightness of the
    bright band at the bottom of the retina (which oct3d.py already found)
    gives a picture "seen from above" where each vessel is a dark line.

Both pictures are turned into a "vessel map" (how much darker than its
surroundings each pixel is). Then we search for the placement of the small OCT
map inside the photo's map that fits best: every size (scale) and turn
(rotation) on a grid, and for each of them EVERY shift at once, by fast
correlation (FFT). The fit is the normalized cross-correlation (NCC):
+1 = the two vessel patterns agree perfectly, 0 = unrelated.

What is searched: size, turn and shift (a "similarity" transform). NOT searched:
mirror images, and the gentle bending between a wide-angle photo (a curved eye
flattened by the camera) and the scan. Inside a 6 mm scan that bending is small,
but it means the alignment is good to roughly a vessel width, not to a pixel.

HONESTY RULES built in:
  - The search always finds SOME best position, even for a photo of another eye.
    So it is run again with deliberately WRONG vessel maps (the scan's map turned
    upside down, and mirrored along its diagonal), which cannot truly match. Their
    best score is the "control": what chance alone reaches. The result is TRUSTED
    only if the real score is at least 1.4 times the control (and at least 0.25).
    Otherwise oct3d.py refuses to fuse unless told to.
  - All of this is hand-written classical image processing, nothing learned,
    and so far checked only on a synthetic photo + scan pair with a known answer.
"""

import math

import numpy as np
from PIL import Image

from eye3d import blurred, eye_area

WORK = 384            # the photo is searched at this size (pixels)
GRID = 160            # the OCT vessel map is made at GRID x GRID pixels over the scanned square


def photo_vessel_map(rgb):
    """How vessel-like each photo pixel is: 'darker than the neighbourhood' in the green
    channel (positive part), zero outside the eye and in a thin band along its edge."""
    size = rgb.shape[0]
    green = rgb[..., 1].astype(np.float32)
    dark = np.maximum(blurred(green, size / 40) - blurred(green, 1.0), 0)
    inside = blurred(eye_area(rgb).astype(np.float32) * 255, size / 40) > 250      # the eye, minus its rim
    return dark * inside


def oct_shadow_picture(vol, low, spacing, band_um=80.0):
    """The scan seen from above, made only from the bright band at the bottom of the retina:
    mean brightness from the lower surface to `band_um` below it. Vessels show as dark lines
    (their shadows). Returns [scan, width] floats."""
    k = max(2, round(band_um / spacing["dz"]))
    rows = np.clip(np.round(low).astype(np.int64)[:, None, :] + np.arange(k)[None, :, None], 0, vol.shape[1] - 1)
    return np.take_along_axis(vol, rows, axis=1).astype(np.float32).mean(axis=1)


def oct_vessel_map(shadow_picture):
    """The shadow picture stretched to GRID x GRID (so one pixel is the same size in both
    directions: B-scans are much further apart than the pixels inside one), then
    'darker than the neighbourhood', as for the photo."""
    lo, hi = np.percentile(shadow_picture, [1, 99])
    img = Image.fromarray(np.clip((shadow_picture - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8))
    even = np.asarray(img.resize((GRID, GRID), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.maximum(blurred(even, GRID / 16) - blurred(even, 1.0), 0)


def window_sums(image, h, w):
    """Sum of `image` inside every h x w window (top-left corner at each position), by a
    summed-area table: 4 look-ups per window instead of h * w additions."""
    table = np.pad(image.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]


def ncc_map(image, image_fft, template):
    """
    Normalized cross-correlation of `template` with `image` at every shift where the template
    lies fully inside the image. For one shift:
        NCC = sum((I - mean I) * (T - mean T)) / sqrt(sum((I - mean I)^2) * sum((T - mean T)^2))
    over the window. The numerator for ALL shifts at once is one correlation, done with the
    FFT (correlating in space = multiplying in frequency); the image part of the denominator
    comes from window_sums.
    """
    h, w = template.shape
    t = template - template.mean()
    padded = np.zeros_like(image, dtype=np.float32)
    padded[:h, :w] = t
    numer = np.fft.irfft2(image_fft * np.conj(np.fft.rfft2(padded)), s=image.shape)[:image.shape[0] - h + 1,
                                                                                    :image.shape[1] - w + 1]
    s1, s2 = window_sums(image, h, w), window_sums(image.astype(np.float64) ** 2, h, w)
    var = np.maximum(s2 - s1 ** 2 / (h * w), 0)
    # Windows with almost no vessels in them (flat) get NCC 0 instead of a noisy division.
    floor = 0.02 * var.max() + 1e-8
    return np.where(var > floor, numer / np.sqrt(np.maximum(var, floor) * (t ** 2).sum() + 1e-12), 0.0)


def placed_template(oct_map, size_px, angle_deg):
    """The OCT vessel map at a candidate size (pixels in the photo) and turn (degrees,
    counter-clockwise as seen on screen); corners uncovered by the turn are filled with the average."""
    img = Image.fromarray(np.clip(oct_map / (oct_map.max() + 1e-8) * 255, 0, 255).astype(np.uint8))
    img = img.resize((size_px, size_px), Image.Resampling.BILINEAR)
    fill = int(np.asarray(img).mean())
    return np.asarray(img.rotate(angle_deg, Image.Resampling.BILINEAR, fillcolor=fill), dtype=np.float32)


def register(photo_rgb, oct_map, oct_width_mm, photo_width_mm=13.0, scales=None, angles=None,
             min_ncc=0.25, min_ratio=1.4):
    """
    Find where the OCT square lies in the photo. `photo_width_mm` is a first guess of how
    much retina the photo's width covers (about 13 mm for a 45-degree camera); the search
    tries sizes from 0.6x to 1.6x of what that guess implies, and turns of -16..+16 degrees,
    coarse first, then fine around the best.
    Returns a dict: `matrix` (2x3: scan position (u, v) in 0..1 -> photo position (x, y) in
    0..1), the size/turn found, `ncc`, `control` (best score of the wrong maps) and `trusted`.
    """
    photo = np.asarray(Image.fromarray(photo_rgb).resize((WORK, WORK), Image.Resampling.BILINEAR))
    vessels = photo_vessel_map(photo)
    vessels_fft = np.fft.rfft2(vessels)
    guess_px = oct_width_mm / photo_width_mm * WORK            # the scan's width in photo pixels, if the guess is right

    def try_all(scale_list, angle_list, oct_map=oct_map):
        best = None
        for s in scale_list:
            size = int(round(guess_px * s))
            if size < 24 or size > WORK - 8:
                continue
            for a in angle_list:
                scores = ncc_map(vessels, vessels_fft, placed_template(oct_map, size, a))
                y, x = np.unravel_index(scores.argmax(), scores.shape)
                if best is None or scores[y, x] > best["ncc"]:
                    best = {"ncc": float(scores[y, x]), "scale": float(s), "angle": float(a), "size": size,
                            "x": int(x), "y": int(y),
                            # how far the peak stands above the other positions for this size and turn
                            "standout": float((scores[y, x] - scores.mean()) / (scores.std() + 1e-8))}
        return best

    scales = scales if scales is not None else np.exp(np.linspace(math.log(0.6), math.log(1.6), 21))
    angles = angles if angles is not None else np.arange(-16, 17, 4)
    coarse = try_all(scales, angles)
    if coarse is None:
        raise SystemExit("Registration: the scan is too small or too large for the photo at every size tried; "
                         "check --photo-width-mm.")
    best = try_all(coarse["scale"] * np.linspace(0.95, 1.05, 11), coarse["angle"] + np.arange(-3, 4, 1.0))
    if best["ncc"] < coarse["ncc"]:
        best = coarse
    # The control: the same search with maps that cannot be right.
    wrong = [try_all(scales, angles, m) for m in (oct_map[::-1, ::-1], oct_map.T.copy())]
    control = max(w["ncc"] for w in wrong if w is not None)

    # The transform, built the same way the template was: stretch the scan square to `size`
    # pixels, turn it about its centre, then shift it to (x, y). A screen has y pointing DOWN,
    # so a counter-clockwise turn on screen by angle a moves (dx, dy) to
    # (dx cos a + dy sin a, -dx sin a + dy cos a).
    a, size, half = math.radians(best["angle"]), best["size"], best["size"] / 2
    cos, sin = math.cos(a), math.sin(a)
    #   photo_px = (x, y) + half + R * (u * size - half, v * size - half)
    matrix = np.array([[cos * size, sin * size, best["x"] + half - (cos + sin) * half],
                       [-sin * size, cos * size, best["y"] + half - (cos - sin) * half]]) / WORK
    best.update({
        "matrix": matrix.tolist(),
        "photo_mm_per_width": float(oct_width_mm * WORK / size),     # what the photo's width covers, as found
        "control": float(control), "ratio": float(best["ncc"] / max(control, 1e-6)),
        "trusted": bool(best["ncc"] >= min_ncc and best["ncc"] >= min_ratio * control),
        "min_ncc": min_ncc, "min_ratio": min_ratio,
    })
    return best


def scan_to_photo(matrix, u, v):
    """Scan position (u, v in 0..1) -> photo position (x, y in 0..1)."""
    m = np.asarray(matrix)
    return m[0, 0] * u + m[0, 1] * v + m[0, 2], m[1, 0] * u + m[1, 1] * v + m[1, 2]


def warp_to_scan(picture, matrix, out_size=512):
    """
    The part of a photo-frame picture (the photo itself, or its heat map) that lies over the
    scan, resampled onto the scan's own square: out[v, u] = picture at scan_to_photo(u, v),
    blended from the 4 nearest pixels (bilinear). Points outside the photo come out 0.
    """
    pic = np.asarray(picture, dtype=np.float32)
    h, w = pic.shape[:2]
    grid = (np.arange(out_size) + 0.5) / out_size
    u, v = np.meshgrid(grid, grid)
    x, y = scan_to_photo(matrix, u, v)
    x, y = x * w - 0.5, y * h - 0.5
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0, y0 = np.clip(np.floor(x), 0, w - 2).astype(int), np.clip(np.floor(y), 0, h - 2).astype(int)
    ax, ay = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    if pic.ndim == 3:
        ax, ay, inside = ax[..., None], ay[..., None], inside[..., None]
    out = ((pic[y0, x0] * (1 - ax) + pic[y0, x0 + 1] * ax) * (1 - ay)
           + (pic[y0 + 1, x0] * (1 - ax) + pic[y0 + 1, x0 + 1] * ax) * ay)
    return np.where(inside, out, 0).astype(np.float32)


def check_picture(photo_rgb, shadow_picture, matrix, size=320):
    """
    A picture to judge the alignment by eye, two panels side by side:
      left:  the photo with the scan's outline drawn where the registration put it;
      right: a checkerboard over the scanned square, alternating tiles of the photo (green
             channel) and of the OCT shadow picture. If the alignment is right, vessels run
             straight across the tile borders; if not, they break at every border.
    """
    from PIL import ImageDraw
    left = Image.fromarray(photo_rgb).resize((size, size), Image.Resampling.BILINEAR)
    corners = [scan_to_photo(matrix, u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    ImageDraw.Draw(left).polygon([(x * size, y * size) for x, y in corners], outline=(90, 200, 255))

    def stretch(a):
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / (hi - lo + 1e-8) * 255, 0, 255)
    from_photo = stretch(warp_to_scan(photo_rgb[..., 1], matrix, size))
    from_oct = stretch(np.asarray(Image.fromarray(stretch(shadow_picture).astype(np.uint8))
                                  .resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32))
    tile = size // 6
    yy, xx = np.mgrid[:size, :size]
    board = np.where(((yy // tile) + (xx // tile)) % 2 == 0, from_photo, from_oct).astype(np.uint8)
    out = Image.new("RGB", (2 * size + 8, size), (14, 17, 22))
    out.paste(left, (0, 0))
    out.paste(Image.fromarray(board).convert("RGB"), (size + 8, 0))
    return np.asarray(out)
