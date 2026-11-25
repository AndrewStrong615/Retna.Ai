"""
convert_olives.py: OLIVES (Prabhushankar et al. 2022, Zenodo 7105232, CC BY 4.0) -> a pairs
folder for `photo2oct.py train`.

What OLIVES holds (seen in the files themselves, 2026-11-25)
-----------------------------------------------------------
OLIVES.zip contains two zips: Prime_FULL.zip (PRIME trial, diabetic retinopathy) and
TREX_DME.zip (TREX-DME trial, diabetic macular edema). Inside, one folder per patient, visit
and eye, e.g. `Prime_FULL/01-001/W0/OD/`, holding
  - 49 B-scans (PRIME: `0.png` ... `48.png` or .tif; TREX-DME: `TREXW_000000.tif` ... or .jpg; 504 x 496 px), a Heidelberg Spectralis scan of
    about 6 x 6 mm (20 x 20 degrees) around the macula; the burnt-in scale bar says
    200 um = 51 px in depth (3.9 um per pixel) and ~16.5 px sideways (~12 um per pixel);
  - one fundus picture (`fundus_OD_W0.tif`, 768 x 768 px, ~9 mm = 30 degrees wide). It is a
    NEAR-INFRARED picture taken by the OCT machine itself (grey, vessels dark): NOT a colour
    photo. A network trained on these has never seen a colour photo.

What this script does per folder (one folder = one pair)
--------------------------------------------------------
  1. THICKNESS: oct3d.find_layers finds the retina's inner surface (ILM) and the top of the
     bright photoreceptor/RPE band in all 49 B-scans; the distance is the thickness map.
     Spectralis itself measures a little deeper (down to Bruch's membrane, under that band).
     So a CONSTANT is added to every map: the median difference between the machine's own
     central thickness (`CST`, in OLIVES' Clinical_Data_Images table) and ours over the central
     1 mm, taken over all visits that have a label. The fit is printed and saved, and it is
     also the quality check of the whole measurement: if our central thickness did not follow
     the machine's, the maps would be wrong.
  2. WHERE THE SCAN LIES IN THE PICTURE: the machine scans the middle of its own picture:
     the scan square is the central 2/3 (20 of 30 degrees), and B-scan 0 is the BOTTOM row
     (Spectralis numbers from below). Both were found, not assumed: register.py's vessel
     matching put the square at (0.50, 0.50), size 1.02 x that guess, turn 0, only with the
     scan order reversed. Every folder is checked the same way (`--check-every`); where the
     match is TRUSTED its own placement is used, otherwise the central 2/3.
  3. The picture's square is cut out at 128 px, the map resized to 64 x 64, both mirrored for
     right eyes (OD) so the optic disc side is always on the LEFT, as photo2oct.py expects.
  4. A scan whose surfaces were clearly not found (see `plausible`) is left out and counted.

Writes <pairs>/<name>.png + <name>.npy + pairs.csv (name, patient) like photo2oct.add_pair,
plus pairs.json (source, counts, the CST fit, `"input": "grey"`: tells `train` that these
pictures have no colour, so colour photos must be turned grey before the network sees them)
and olives_table.csv (one row per folder with every number used).

The patient is the trial's patient number (both eyes and all visits of a person share it), so
`photo2oct.py train` keeps a person's eyes and visits in ONE split.

Usage (the zips are read in place, nothing is unpacked):
  python convert_olives.py --src Prime_FULL.zip TREX_DME.zip --clinical Clinical_Data_Images.xlsx --pairs pairs_olives
"""

import argparse
import csv
import io
import json
import multiprocessing
import os
import re
import zipfile

import numpy as np
from PIL import Image

import oct3d
import photo2oct as p2o
import register as reg

SCAN_MM = 6.0                 # the scan square: 20 degrees, about 6 mm
SCAN_SHARE = 2.0 / 3.0        # ... of the 30-degree fundus picture's width
DZ_UM = 200.0 / 51.0          # depth per pixel, from the burnt-in scale bar (200 um = 51 px)


class Source:
    """A zip file or a plain folder, read the same way: names() and read(name)."""

    def __init__(self, path):
        self.path, self.zip = path, None
        if os.path.isdir(path):
            self.all = [os.path.relpath(os.path.join(d, f), path).replace(os.sep, "/")
                        for d, _, files in os.walk(path) for f in files]
        else:
            self.zip = zipfile.ZipFile(path)
            self.all = [n for n in self.zip.namelist() if not n.endswith("/")]

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        with open(os.path.join(self.path, name), "rb") as f:
            return f.read()


def scan_folders(source):
    """
    Group a source's files by folder and keep the folders that are one scan: exactly 49
    B-scan pictures (file names ending in a number) + a fundus picture + an eye (OD / OS)
    somewhere in the path. Returns a list of dicts and a list of (folder, reason) left out.
    """
    by_folder = {}
    for name in source.all:
        folder, _, base = name.rpartition("/")
        if base.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")):
            by_folder.setdefault(folder, []).append(base)
    found, skipped = [], []
    for folder, files in sorted(by_folder.items()):
        fundus = sorted(f for f in files if "fundus" in f.lower())
        # B-scans: files named <prefix><number>.<ext>. Grouped by prefix and zero-padding, because TREX-DME
        # folders also hold one more picture named like a B-scan with fewer digits (TREXW_001.tif beside
        # TREXW_000000.tif ... TREXW_000048.tif). The scan is the group of exactly 49 (else the largest, to report).
        groups = {}
        for f in files:
            m = re.fullmatch(r"(.*?)(\d+)\.[a-zA-Z]+", f)
            if m and "fundus" not in f.lower():
                digits = m.group(2)
                padded = len(digits) if len(digits) > 1 and digits[0] == "0" else 0     # 0 = written without leading zeros
                groups.setdefault((m.group(1), padded), {})[int(digits)] = f
        scans = next((g for g in groups.values() if len(g) == 49), max(groups.values(), key=len, default={}))
        parts = folder.split("/")
        eye = next((p for p in reversed(parts) if p in ("OD", "OS")), None)
        if not fundus or eye is None or len(scans) != 49:
            if fundus or len(scans) >= 10:
                skipped.append((folder, f"{len(scans)} B-scans, {len(fundus)} fundus pictures, eye {eye}"))
            continue
        # Prefer the fundus file that names the eye (some folders hold two copies of the same picture).
        named = [f for f in fundus if f"_{eye}_" in f or f"_{eye}." in f]
        found.append({"folder": folder, "eye": eye, "fundus": (named or fundus)[0],
                      "scans": [scans[k] for k in sorted(scans)]})
    return found, skipped


def patient_of(folder):
    """
    The person a folder belongs to, from the path. PRIME: `Prime_FULL/01-001/W0/OD` -> prime_01-001.
    TREX-DME: `TREX DME/GILA/0201GOD/V1/OD` -> trex_0201 (the 4 digits are the patient; the letters
    after them are the treatment arm and the eye). Anything else: the folder two levels above the eye.
    """
    parts = folder.split("/")
    top = parts[0].lower()
    if "prime" in top and len(parts) >= 2:
        return "prime_" + parts[1]
    for p in parts:
        m = re.fullmatch(r"(\d{4})[A-Z]{1,2}O[DS]", p)
        if m:
            return "trex_" + m.group(1)
    return "_".join(parts[:-2]) or parts[0]


def read_volume(source, item):
    """The 49 B-scans as [49, depth, width] uint8 (in the files' own order: scan 0 first)."""
    pics = [np.asarray(Image.open(io.BytesIO(source.read(f"{item['folder']}/{f}"))).convert("L")) for f in item["scans"]]
    if len({p.shape for p in pics}) != 1:
        raise ValueError("B-scans of different sizes")
    return np.stack(pics)


def plausible(thick_um):
    """
    Were the two surfaces found? A retina is roughly 150-700 um thick (far more only in heavy
    swelling). The finder fails on a scan that is cut off at the top, very dark or folded over:
    then large parts of the map are far too thin or too thick. Returns (ok, reason).
    """
    too_thin, too_thick = float((thick_um < 90).mean()), float((thick_um > 900).mean())
    if too_thin > 0.05:
        return False, f"{too_thin:.0%} of the map under 90 um"
    if too_thick > 0.05:
        return False, f"{too_thick:.0%} of the map over 900 um"
    return True, ""


def nominal_matrix():
    """Scan square (u, v in 0..1) -> picture (0..1): the central 2/3, no turn. The machine's own geometry."""
    k, off = SCAN_SHARE, (1 - SCAN_SHARE) / 2
    return [[k, 0.0, off], [0.0, k, off]]


def measure(source, item, check=True):
    """
    One folder -> (crop uint8 [128, 128, 3], thickness um [64, 64] WITHOUT the constant, info dict).
    The volume is flipped to top-row-first right away, so everything after is in picture order.
    """
    vol = read_volume(source, item)[::-1]                                # B-scan 0 is the bottom row
    spacing = {"dx": SCAN_MM * 1000 / vol.shape[2], "dy": SCAN_MM * 1000 / vol.shape[0], "dz": DZ_UM}
    ilm, low = oct3d.find_layers(vol, spacing)
    thick = (low - ilm) * DZ_UM                                          # [49, width] um
    ok, why = plausible(thick)
    fundus = np.asarray(Image.open(io.BytesIO(source.read(f"{item['folder']}/{item['fundus']}"))).convert("L"))
    rgb = np.stack([fundus] * 3, axis=-1)                                # grey, as 3 equal channels
    matrix, info = nominal_matrix(), {"ok": ok, "why": why, "placement": "central 2/3"}
    if check and ok:
        # The vessel check: only sizes and turns close to the machine's geometry are tried.
        found = reg.register(rgb, reg.oct_vessel_map(reg.oct_shadow_picture(vol, low, spacing)), SCAN_MM,
                             photo_width_mm=SCAN_MM / SCAN_SHARE, scales=np.linspace(0.94, 1.06, 7),
                             angles=np.arange(-3, 4, 1.5))
        cx, cy = reg.scan_to_photo(found["matrix"], 0.5, 0.5)
        info.update({"ncc": found["ncc"], "control": found["control"], "trusted": found["trusted"],
                     "centre_x": float(cx), "centre_y": float(cy), "scale": found["scale"], "angle": found["angle"]})
        if found["trusted"]:
            matrix, info["placement"] = found["matrix"], "vessel match"
    nasal = "right" if item["eye"] == "OD" else "left"                   # a right eye's disc is on the picture's right
    crop = p2o.macula_crop(rgb, matrix, nasal)
    small = np.asarray(Image.fromarray(thick.astype(np.float32), mode="F")
                       .resize((p2o.OUT_PX, p2o.OUT_PX), Image.Resampling.BILINEAR))
    if nasal == "right":
        small = small[:, ::-1]
    # Our central thickness: mean inside the 1 mm circle at the square's centre (the machine's CST is the same circle
    # at the fovea; the two agree when the scan was centred on the fovea, as the operators aim to).
    c = (np.arange(p2o.OUT_PX) + 0.5) / p2o.OUT_PX * SCAN_MM - SCAN_MM / 2
    x, y = np.meshgrid(c, c)
    info["central_um"] = float(small[np.hypot(x, y) < 0.5].mean())
    return crop, small.copy(), info


def read_cst(path):
    """OLIVES' Clinical_Data_Images (.xlsx or .csv: File_Path, BCVA, CST, Eye_ID, Patient_ID) -> {folder path: CST um}.
    Paths are compared without their top folder and without case/space differences (the table writes
    '/Prime_FULL/01-001/W0/OS/...', '/TREX DME/GILA/...')."""
    if path.lower().endswith(".xlsx"):
        import openpyxl
        rows = openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True)
    else:
        rows = csv.reader(open(path, newline=""))
    out = {}
    for i, r in enumerate(rows):
        if i == 0 or not r or r[0] is None:
            continue
        try:
            out[folder_key(str(r[0]).rpartition("/")[0])] = float(r[2])
        except (TypeError, ValueError):
            pass
    return out


def folder_key(folder):
    """A folder path reduced to what both the zip and the table agree on: its parts after the top folder."""
    parts = [p for p in folder.replace("\\", "/").split("/") if p]
    return "/".join(parts[1:]).lower().replace(" ", "")


_SOURCE = None                # each worker process's own open source


def _open_source(path):
    global _SOURCE
    _SOURCE = Source(path)


def _one_folder(job):
    """One folder -> its row (run inside a worker process)."""
    item, check = job
    row = {"folder": item["folder"], "eye": item["eye"], "patient": patient_of(item["folder"]),
           "name": re.sub(r"[^A-Za-z0-9]+", "_", item["folder"]).strip("_")}
    try:
        crop, thick, info = measure(_SOURCE, item, check)
        row.update(info)
        row["crop"], row["thick"] = crop, thick
    except Exception as e:                                               # a broken picture must not stop 1,500 others
        row.update({"ok": False, "why": f"unreadable: {e}"})
    return row


def main():
    p = argparse.ArgumentParser(description="OLIVES -> pairs folder for photo2oct.py train")
    p.add_argument("--src", nargs="+", required=True, help="Prime_FULL.zip and/or TREX_DME.zip (or unpacked folders)")
    p.add_argument("--clinical", help="OLIVES' Clinical_Data_Images.xlsx/.csv: the machine's central thickness (CST), "
                                      "for the constant and the quality check")
    p.add_argument("--pairs", required=True)
    p.add_argument("--check-every", type=int, default=1, help="run the vessel check on every Nth folder (0 = never)")
    p.add_argument("--limit", type=int, default=0, help="only the first N folders of each source (for a quick try)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="processes working side by side")
    args = p.parse_args()
    if os.path.exists(os.path.join(args.pairs, "pairs.csv")):
        raise SystemExit(f"{args.pairs} already holds pairs. Pick a new --pairs.")
    cst = read_cst(args.clinical) if args.clinical else {}
    rows = []
    for src in args.src:
        found, skipped = scan_folders(Source(src))
        print(f"{src}: {len(found)} scan folders, {len(skipped)} other folders left out"
              + "".join(f"\n  left out {f}: {why}" for f, why in skipped[:5]), flush=True)
        if args.limit:
            found = found[:args.limit]
        jobs = [(item, bool(args.check_every) and i % args.check_every == 0) for i, item in enumerate(found)]
        # Several processes, each with its own open copy of the source (a zip handle cannot be shared).
        with multiprocessing.Pool(args.workers, initializer=_open_source, initargs=(src,)) as pool:
            for row in pool.imap(_one_folder, jobs, chunksize=4):
                row["cst_label"] = cst.get(folder_key(row["folder"]))
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"  {len(rows)} folders done", flush=True)
    write(rows, args.pairs, args.src)


def write(rows, pairs, sources):
    """The second half: the constant from the CST labels, then the pairs folder."""
    good = [r for r in rows if r["ok"]]
    if not good:
        raise SystemExit("No usable scan found.")
    labelled = [r for r in good if r.get("cst_label")]
    fit = {"labelled": len(labelled)}
    offset = 0.0
    if len(labelled) >= 20:
        ours, theirs = np.array([r["central_um"] for r in labelled]), np.array([r["cst_label"] for r in labelled])
        offset = float(np.median(theirs - ours))
        left = theirs - ours - offset
        fit.update({"offset_um": offset, "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
                    "median_abs_difference_um": float(np.median(np.abs(left))),
                    "share_within_25um": float((np.abs(left) < 25).mean())})
        print(f"CST check on {len(labelled)} labelled visits: the machine's central thickness = ours + {offset:.0f} um "
              f"(median); correlation {fit['correlation']:.3f}; after the constant, median difference "
              f"{fit['median_abs_difference_um']:.0f} um, {fit['share_within_25um']:.0%} within 25 um")
    else:
        print("No CST labels: the maps stay 'inner surface to the top of the photoreceptor band' (thinner than the "
              "machine's own thickness by a constant of roughly 60-80 um).")
    checked = [r for r in good if "trusted" in r]
    trusted = [r for r in checked if r["trusted"]]
    place = {"checked": len(checked), "trusted": len(trusted)}
    if trusted:
        off = np.array([np.hypot(r["centre_x"] - 0.5, r["centre_y"] - 0.5) for r in trusted])
        place.update({"median_centre_offset": float(np.median(off)), "share_within_3pct": float((off < 0.03).mean())})
        print(f"Placement check: {len(trusted)} of {len(checked)} vessel matches trusted; among them the scan's centre "
              f"lies a median {place['median_centre_offset']:.3f} of the picture's width from the middle, "
              f"{place['share_within_3pct']:.0%} within 0.03")
    for r in good:
        p2o.add_pair(pairs, r["name"], r["patient"], r["crop"], r["thick"] + offset)
    with open(os.path.join(pairs, "pairs.json"), "w") as f:
        json.dump({"source": "OLIVES (Prabhushankar et al. 2022, Zenodo 7105232, CC BY 4.0): near-infrared fundus + Spectralis OCT",
                   "files": [os.path.basename(s) for s in sources], "input": "grey",
                   "pairs": len(good), "patients": len({r["patient"] for r in good}),
                   "left_out": len(rows) - len(good), "cst_fit": fit, "placement": place,
                   "thickness": "ILM to the top of the photoreceptor/RPE band (oct3d.find_layers) + offset_um"}, f, indent=2)
    keys = ["name", "folder", "patient", "eye", "ok", "why", "placement", "ncc", "control", "trusted", "centre_x", "centre_y",
            "scale", "angle", "central_um", "cst_label"]
    with open(os.path.join(pairs, "olives_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(good)} pairs of {len({r['patient'] for r in good})} patients written to {pairs}; {len(bad)} scans left out"
          + "".join(f"\n  left out {r['folder']}: {r['why']}" for r in bad[:8]))


if __name__ == "__main__":
    main()
