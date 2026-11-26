"""
Turn DeepDRiD's regular fundus photos into flat photo folders plus one labels CSV this
project's scripts can read. DeepDRiD is a FRESH external clinic for the set-up study
(PREREG.md, E1-E3): never trained on, never used to choose anything.

DeepDRiD (Liu et al., Patterns 2022; licence CC BY-SA 4.0; cite the paper): 2,000 regular
fundus photos of 500 patients from the Sixth People's Hospital, Shanghai. Every eye has TWO
photos (`<patient>_l1`, `_l2`, `_r1`, `_r2`), and every photo has a DR grade (0-4) and an
overall quality label (0 = low, 1 = high; plus clarity, field definition and artifact scores).
It ships in three parts, each a folder per patient:

    regular-fundus-training/      300 patients, 1,200 photos   -> part "train"
    Online-Challenge1&2-Evaluation 100 patients, 400 photos    -> part "eval" (grades in xlsx)
    regular-fundus-validation/    100 patients, 400 photos     -> part "sealed"

The challenge's own split means nothing here (no model trains on DeepDRiD). "train" and
"eval" together are the clinic for E1-E3; "sealed" is set aside from day one for the single
end-to-end run of the finished screener (plan step S5) and must not be scored before then.

Output (--out):
    photos/<part>/dd<patient>_<view>.jpg   copies, one flat folder per part (preprocess.py --src)
    deepdrid_labels.csv                    image, level, patient, eye, view, part, quality,
                                           clarity, field_definition, artifact
    deepdrid_<part>.csv                    image, level only (the plain format external_test.py reads)
Names start with "dd" + the patient number, so the patient is everything before the first "_"
(the same rule as EyePACS names), and both photos of both eyes stay together in any split.

Usage:
  python convert_deepdrid.py --repo <DeepDRiD clone> --out <retina-data>/deepdrid
"""

import argparse
import os
import shutil

import pandas as pd

PARTS = {"train": "regular-fundus-training", "eval": "Online-Challenge1&2-Evaluation",
         "sealed": "regular-fundus-validation"}


def find_photos(folder):
    """Every .jpg under a part's Images folder (one sub-folder per patient), by image_id."""
    found = {}
    for root, _, files in os.walk(os.path.join(folder, "Images")):
        for f in files:
            if f.lower().endswith(".jpg"):
                found[os.path.splitext(f)[0]] = os.path.join(root, f)
    return found


def read_part(base, part):
    folder = os.path.join(base, PARTS[part])
    if part == "eval":
        grades = pd.read_excel(os.path.join(folder, "Challenge1_labels.xlsx"))      # image_id, DR_Levels
        quality = pd.read_excel(os.path.join(folder, "Challenge2_labels.xlsx"))
        df = grades.rename(columns={"DR_Levels": "level"}).merge(quality, on="image_id", how="left")
    else:
        df = pd.read_csv(os.path.join(folder, f"{PARTS[part]}.csv"))
        # Each row is one photo of one eye: the other eye's grade column is empty.
        df["level"] = df["left_eye_DR_Level"].fillna(df["right_eye_DR_Level"])
    df = df.rename(columns={"Overall quality": "quality", "Clarity": "clarity",
                            "Field definition": "field_definition", "Artifact": "artifact"})
    df["image_id"] = df["image_id"].astype(str)
    df["patient"] = "dd" + df["image_id"].str.split("_").str[0]
    df["eye"] = df["image_id"].str.split("_").str[1].str[0].map({"l": "left", "r": "right"})
    df["view"] = df["image_id"].str.split("_").str[1].str[1:]
    df["image"] = "dd" + df["image_id"]
    df["part"] = part
    photos = find_photos(folder)
    df["source"] = df["image_id"].map(photos)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, help="the DeepDRiD clone (holds regular_fundus_images)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    base = os.path.join(args.repo, "regular_fundus_images")
    if os.path.exists(os.path.join(args.out, "deepdrid_labels.csv")):
        raise SystemExit(f"{args.out} already holds a conversion; use a new --out.")

    df = pd.concat([read_part(base, part) for part in PARTS], ignore_index=True)
    missing = df["source"].isna() | df["level"].isna()
    if missing.any():
        print(f"Left out {int(missing.sum())} rows without a photo or a grade: {df.loc[missing, 'image'].tolist()[:10]}")
    df = df[~missing].copy()
    df["level"] = df["level"].astype(int)
    if not df["level"].between(0, 4).all():
        raise SystemExit("Grades outside 0-4 found.")
    if df["image"].duplicated().any():
        raise SystemExit("Duplicate photo names across parts.")
    if df.groupby("patient")["part"].nunique().max() > 1:
        raise SystemExit("A patient appears in two parts.")

    for part in PARTS:
        dst = os.path.join(args.out, "photos", part)
        os.makedirs(dst, exist_ok=True)
        rows = df[df["part"] == part]
        for name, src in zip(rows["image"], rows["source"]):
            shutil.copyfile(src, os.path.join(dst, name + ".jpg"))
        rows[["image", "level"]].to_csv(os.path.join(args.out, f"deepdrid_{part}.csv"), index=False)
        mix = rows["level"].value_counts().sort_index().to_dict()
        print(f"{part:6s}: {len(rows)} photos, {rows['patient'].nunique()} patients, grades {mix}, "
              f"referable {(rows['level'] >= 2).mean():.1%}, high quality {(rows['quality'] == 1).mean():.1%}")
    cols = ["image", "level", "patient", "eye", "view", "part", "quality", "clarity", "field_definition", "artifact"]
    df[cols].to_csv(os.path.join(args.out, "deepdrid_labels.csv"), index=False)
    print(f"Wrote {args.out}. The 'sealed' part is for the final screener run only: do not score it before then.")


if __name__ == "__main__":
    main()
