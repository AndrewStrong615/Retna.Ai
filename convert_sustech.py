"""
Turn the SUSTech-SYSU fundus dataset into one flat photo folder plus a labels CSV (image,level).
A FRESH external clinic for the set-up study (PREREG.md): never trained on, never used to
choose anything.

Dataset: "The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy
grading" (Lin et al., Scientific Data 2020; figshare article 12570770, licence CC BY 4.0;
cite the paper). 1,219 colour photos (Shenzhen, China), DR graded on the international 0-4
scale, plus a grade "5" = "DR with laser spots or scars". The dataset's second file,
c5_DR_reclassified.csv, re-grades every grade-5 photo on the 0-4 scale; those re-grades are
used here (all are 2-4, so referable either way).
Left/right eye is given, but no patient number, so each photo is its own patient (names
carry no "_"; same as APTOS, IDRiD and Paraguay).

Usage:
  python convert_sustech.py --zip <the figshare zip> --out <retina-data>/sustech
"""

import argparse
import io
import os
import zipfile

import pandas as pd

LABELS = "originalImages/drLabels.csv"
REGRADED = "originalImages/c5_DR_reclassified.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    labels_path = os.path.join(args.out, "sustech_labels.csv")
    if os.path.exists(labels_path):
        raise SystemExit(f"{labels_path} exists; use a new --out.")
    with zipfile.ZipFile(args.zip) as z:
        d = pd.read_csv(io.BytesIO(z.read(LABELS)))
        c = pd.read_csv(io.BytesIO(z.read(REGRADED)))
        # Columns by position (the headers are long): file, eye (0 left / 1 right), ICDR grade.
        d = d.iloc[:, :3].set_axis(["file", "eye", "grade"], axis=1)
        regrade = dict(zip(c.iloc[:, 0], c.iloc[:, 1]))
        laser = d["grade"] == 5
        missing = set(d.loc[laser, "file"]) - set(regrade)
        if missing:
            raise SystemExit(f"Grade-5 photos without a re-grade: {sorted(missing)[:5]}")
        d["level"] = [regrade[f] if g == 5 else g for f, g in zip(d["file"], d["grade"])]
        if not d["level"].between(0, 4).all():
            raise SystemExit("Grades outside 0-4 after re-grading.")
        photos = os.path.join(args.out, "photos")
        os.makedirs(photos, exist_ok=True)
        rows = []
        for f, eye, lvl, g in zip(d["file"], d["eye"], d["level"], d["grade"]):
            stem = os.path.splitext(f)[0]
            name = f"su{stem}"                  # e.g. su0001: plain, no "_" or path parts
            if not name.isalnum():
                raise SystemExit(f"Unexpected file name {f!r}.")
            with open(os.path.join(photos, name + ".jpg"), "wb") as dst:
                dst.write(z.read(f"originalImages/{f}"))
            rows.append({"image": name, "level": int(lvl), "eye": "right" if eye == 1 else "left",
                         "laser_scars": bool(g == 5)})
    df = pd.DataFrame(rows).sort_values("image")
    if df["image"].duplicated().any():
        raise SystemExit("Duplicate names.")
    df.to_csv(labels_path, index=False)
    print(f"{len(df)} photos, grades {df['level'].value_counts().sort_index().to_dict()} "
          f"({df['laser_scars'].sum()} laser-scar photos re-graded), referable {(df['level'] >= 2).mean():.1%}. "
          f"Wrote {labels_path}.")


if __name__ == "__main__":
    main()
