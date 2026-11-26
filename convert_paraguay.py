"""
Turn the Paraguay fundus dataset into one flat photo folder plus a labels CSV (image,level).
A FRESH external clinic for the set-up study (PREREG.md, E1 only): never trained on, never
used to choose anything.

Dataset: "Dataset from fundus images for the study of diabetic retinopathy" (Benítez et al.,
Data in Brief 36, 107068, 2021; Zenodo record 4647952, version 0.2; licence CC BY 4.0; cite
the paper). 757 colour photos from the Department of Ophthalmology, Hospital de Clínicas,
Universidad Nacional de Asunción (Zeiss Visucam 500), one folder per grade on a 7-step scale.
Mapped to the 0-4 international scale used everywhere in this project:

    1. No DR signs             -> 0        4. Severe NPDR       -> 3
    2. Mild (or early) NPDR    -> 1        5. Very Severe NPDR  -> 3
    3. Moderate NPDR           -> 2        6. PDR               -> 4
                                           7. Advanced PDR      -> 4

Only 187 photos are grade 0 and 4 are grade 1, so the set is mostly sick: too few healthy
photos for low disease rates (E2), fine for E1 (the IDRiD repeat at the natural mix).
No patient or eye information: each photo is its own patient (names carry no "_").

Usage:
  python convert_paraguay.py --zip <the Zenodo zip> --out <retina-data>/paraguay
"""

import argparse
import os
import zipfile

import pandas as pd

GRADE = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 3, "6": 4, "7": 4}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    labels_path = os.path.join(args.out, "paraguay_labels.csv")
    if os.path.exists(labels_path):
        raise SystemExit(f"{labels_path} exists; use a new --out.")
    photos = os.path.join(args.out, "photos")
    os.makedirs(photos, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.zip) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 2 or not parts[1].lower().endswith(".jpg"):
                continue
            folder, file = parts
            step = folder.split(".")[0].strip()
            if step not in GRADE:
                raise SystemExit(f"Unknown grade folder {folder!r}.")
            # A plain name without "_" or path parts: written only inside --out.
            name = f"py{step}n{os.path.splitext(os.path.basename(file))[0]}"
            if not name.isalnum():
                raise SystemExit(f"Unexpected file name {info.filename!r}.")
            with z.open(info) as src, open(os.path.join(photos, name + ".jpg"), "wb") as dst:
                dst.write(src.read())
            rows.append({"image": name, "level": GRADE[step], "grade_7_step": int(step)})
    df = pd.DataFrame(rows).sort_values("image")
    if df["image"].duplicated().any():
        raise SystemExit("Duplicate names.")
    df.to_csv(labels_path, index=False)
    print(f"{len(df)} photos, grades {df['level'].value_counts().sort_index().to_dict()}, "
          f"referable {(df['level'] >= 2).mean():.1%}. Wrote {labels_path}.")


if __name__ == "__main__":
    main()
