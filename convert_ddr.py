"""
Turn the DDR dataset's grading labels into one CSV that train.py --extra-labels can read.

DDR (Li et al., 2019; 13,673 color fundus photos from 147 hospitals in 23 Chinese provinces,
many different cameras) ships its DR grades as three text files next to three photo folders:

    DR_grading/train.txt   DR_grading/train/007-0004-000.jpg ...
    DR_grading/valid.txt   DR_grading/valid/...
    DR_grading/test.txt    DR_grading/test/...

Each line is "<file name> <grade>". Grades 0-4 are the usual international scale (none, mild,
moderate, severe, proliferative); grade 5 means "ungradable" (photo too poor to grade) and is
left out here. DDR's own train/valid/test split does not matter to this project: all of DDR is
used for TRAINING only (our validation and test eyes stay pure EyePACS), so all three parts are
written to one file.

DDR has no patient numbers: each photo becomes its own "patient" in train.py (its name has no
"_"). That only matters for the both-eyes blend, which never uses training photos.

Usage:
  python convert_ddr.py --root %USERPROFILE%\\retina-data\\ddr\\DDR-dataset\\DR_grading
         --out %USERPROFILE%\\retina-data\\data\\ddr_labels.csv
It prints the photo count and grade mix per part, and how many were left out and why.
Keep the CSV outside the repo (labels are never committed), and follow DDR's license terms.
"""

import argparse
import os

import pandas as pd

UNGRADABLE = 5
PARTS = ("train", "valid", "test")


def read_part(root, part):
    """One DDR part: (rows of image/level/part, number of label lines whose photo is missing)."""
    labels = os.path.join(root, f"{part}.txt")
    if not os.path.exists(labels):
        raise SystemExit(f"{labels} not found: --root must be the DR_grading folder.")
    rows, missing = [], 0
    with open(labels, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit():
                raise SystemExit(f"{labels} line {line_no}: expected '<file> <grade>', got {line.strip()!r}")
            name, grade = fields[0], int(fields[1])
            if not os.path.exists(os.path.join(root, part, name)):
                missing += 1
                continue
            rows.append({"image": os.path.splitext(name)[0], "level": grade, "part": part})
    return rows, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="DDR's DR_grading folder (with train.txt and train/ etc.)")
    p.add_argument("--out", required=True, help="CSV to write (image,level), outside the repo")
    args = p.parse_args()

    rows, missing = [], {}
    for part in PARTS:
        part_rows, missing[part] = read_part(args.root, part)
        rows += part_rows
    df = pd.DataFrame(rows)
    dupes = df["image"][df["image"].duplicated()].tolist()
    if dupes:
        raise SystemExit(f"{len(dupes)} photo names appear in more than one part (e.g. {dupes[:3]}).")
    bad = df[~df["level"].between(0, UNGRADABLE)]
    if len(bad):
        raise SystemExit(f"Unexpected grades {sorted(bad['level'].unique())}: DDR grades are 0-5.")

    print("DDR grading labels, per part (grade counts 0/1/2/3/4, ungradable = grade 5):")
    for part in PARTS:
        g = df[df["part"] == part]["level"].value_counts().reindex(range(6), fill_value=0)
        print(f"  {part:5s}: {int(g.sum()):5d} photos | grades {'/'.join(str(int(g[i])) for i in range(5))} | "
              f"ungradable {int(g[5])} | label lines without a photo {missing[part]}")
    kept = df[df["level"] != UNGRADABLE]
    mix = kept["level"].value_counts(normalize=True).sort_index()
    print(f"Kept {len(kept)} gradable photos ({(df['level'] == UNGRADABLE).sum()} ungradable left out); "
          f"grade mix {' / '.join(f'{v:.1%}' for v in mix)}; referable (grade >= 2) {(kept['level'] >= 2).mean():.1%}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept[["image", "level"]].to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
