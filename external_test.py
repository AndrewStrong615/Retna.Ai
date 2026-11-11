"""
External validation (Stage 4): score photos from ANOTHER hospital, once, with a frozen system.

Every number so far came from EyePACS (same clinics, cameras and graders). This script
answers the question those numbers cannot: does the system still work on photos it has
never seen anything like? It is built for Messidor-2 (France, different camera, grades
adjudicated by a panel of retina specialists: Krause et al., 2018), but works for any
dataset given in the same format.

Nothing is chosen here. The system is defined by its `<run>_tta` folders, and every
choice is re-derived from their VALIDATION predictions exactly as analyze.py does
(validation_choices): each model's score scale, the fellow-eye weight w, and the cutoff.
Two cutoffs are reported, both fixed on validation BEFORE the external photos are scored:
  - standard: refers 90% of the sick validation eyes (the frozen system's own cutoff);
  - cautious: refers >= 90% of the sick eyes in 95% of bootstrap resamples of the
    validation patients. Both v3 and v4 landed at 88-89% sensitivity on the EyePACS test
    set, because a cutoff picked on one set of eyes moves on new eyes; this one leaves a
    safety margin for that. It refers more people, so specificity is lower.

Rules it enforces:
  - external photos are NEVER used for training or for choosing anything;
  - the external set is scored ONCE: it refuses an --out folder that already has results;
  - the photos must be preprocessed exactly like the system's training photos
    (preprocess.json is compared with each run's config.json).

Patients: Messidor-2 names (e.g. 20051020_43808_0100_PP.png, IM000013.JPG) do not contain
a patient number, so the patient of each photo comes from the pairing file (messidor-2.csv:
one row per patient, columns left;right). That grouping drives the both-eyes blend and the
patient-level 95% ranges.

Usage (Windows, after preprocessing the photos like the training photos):
  python preprocess.py --src %USERPROFILE%\\retina-data\\messidor2\\IMAGES
         --dst %USERPROFILE%\\retina-data\\messidor2\\processed_512_enhanced_tight --size 512 --enhance --tight-mask
  python external_test.py --tta-runs runs\\exp12_tta runs\\exp12b_tta
         --images %USERPROFILE%\\retina-data\\messidor2\\processed_512_enhanced_tight
         --grades %USERPROFILE%\\retina-data\\messidor2\\messidor_data.csv
         --pairs %USERPROFILE%\\retina-data\\messidor2\\messidor-2.csv --out runs\\messidor2_v4
If you use Messidor-2 results, cite Krause et al. (2018) and the Messidor program partners
(see messidor_readme.txt and the ADCIS page).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from analyze import (GRADE_NAMES, apply_choices, auc, bootstrap_weights, range95, sens_spec,
                     validation_choices)
from data import FundusDataset, make_transforms
from model import build_model
from tta import score_eyes

REFERABLE_FROM = 2
RECIPE_KEYS = ("size", "enhance", "tight_mask", "mask_margin")


def cautious_cutoff(sick, score, patients, target_sensitivity, n_boot=2000, seed=0, level=0.05):
    """
    A cutoff with a safety margin, chosen on validation only. For each bootstrap resample
    of the validation PATIENTS, find the cutoff that refers target_sensitivity of the sick
    eyes; then take the 5th-lowest percentile of those cutoffs. A lower cutoff refers more
    eyes, so in 95% of the resamples this one still reaches the target sensitivity.
    """
    cuts = []
    for w in bootstrap_weights(patients, n_boot, seed):
        if (w * sick).sum() == 0 or (w * ~sick).sum() == 0:
            continue        # a resample without sick (or healthy) eyes has no cutoff (tiny sets only)
        _, tpr, thresholds = roc_curve(sick, score, sample_weight=w, drop_intermediate=False)
        cuts.append(float(thresholds[np.where(tpr >= target_sensitivity)[0][0]]))
    if not cuts:
        raise SystemExit("Validation has too few sick eyes for a bootstrap cutoff.")
    return float(np.percentile(cuts, 100 * level))


def load_external(grades_csv, pairs_csv, images):
    """
    One row per gradable photo: image (file name without extension, as preprocess.py
    writes it), patient (from the pairing file), DR grade, DME. Columns are those of the
    Messidor-2 adjudicated grades (Krause et al.) and the ADCIS pairing file.
    """
    grades = pd.read_csv(grades_csv)
    need = {"image_id", "adjudicated_dr_grade", "adjudicated_dme", "adjudicated_gradable"}
    if not need <= set(grades.columns):
        raise SystemExit(f"{grades_csv} needs columns {sorted(need)}, has {list(grades.columns)}.")
    pairs = pd.read_csv(pairs_csv, sep=";")
    pairs.columns = [c.strip().lower() for c in pairs.columns]        # the file has "right " with a space
    if not {"left", "right"} <= set(pairs.columns):
        raise SystemExit(f"{pairs_csv} needs columns left;right, has {list(pairs.columns)}.")
    stem = lambda name: os.path.splitext(str(name).strip())[0]
    patient_of = {}
    for i, row in pairs.iterrows():
        for side in ("left", "right"):
            patient_of[stem(row[side])] = f"P{i:04d}"

    df = pd.DataFrame({"image": grades["image_id"].map(stem),
                       "level": grades["adjudicated_dr_grade"].astype(int),
                       "dme": grades["adjudicated_dme"].astype(int),
                       "gradable": grades["adjudicated_gradable"].astype(int)})
    unpaired = sorted(set(df["image"]) - set(patient_of))
    if unpaired:
        raise SystemExit(f"{len(unpaired)} graded photos are not in the pairing file (e.g. {unpaired[:3]}).")
    df["patient"] = df["image"].map(patient_of)
    n_ungradable = int((df["gradable"] == 0).sum())
    df = df[df["gradable"] == 1].copy()
    missing = df[~df["image"].map(lambda n: os.path.exists(os.path.join(images, f"{n}.jpg")))]
    if len(missing):
        raise SystemExit(f"{len(missing)} gradable photos have no preprocessed image in {images} "
                         f"(e.g. {missing['image'].head(3).tolist()}). Preprocess the whole IMAGES folder.")
    df["sick"] = df["level"] >= REFERABLE_FROM
    # Secondary definition used by many published benchmarks: moderate-or-worse DR OR macular edema.
    df["sick_or_dme"] = df["sick"] | (df["dme"] == 1)
    return df.reset_index(drop=True), n_ungradable, len(pairs)


def source_run(tta_dir):
    """The finished training run a <run>_tta folder was made from."""
    info_path = os.path.join(tta_dir, "tta.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            run = json.load(f).get("run")
        if run and os.path.exists(os.path.join(run, "best.pt")):
            return run
    run = tta_dir.rstrip("/\\")
    if run.endswith("_tta") and os.path.exists(os.path.join(run[:-4], "best.pt")):
        return run[:-4]
    raise SystemExit(f"Cannot find the training run behind {tta_dir} (no tta.json or best.pt).")


def check_recipe(run, images):
    """The external photos must be preprocessed exactly like the run's training photos."""
    path = os.path.join(images, "preprocess.json")
    with open(os.path.join(run, "config.json")) as f:
        trained = json.load(f).get("preprocess")
    if not os.path.exists(path) or not trained:
        print(f"  (could not compare preprocessing for {run}: preprocess.json missing on one side)")
        return
    with open(path) as f:
        ext = json.load(f)
    a, b = ({k: d.get(k) for k in RECIPE_KEYS} for d in (trained, ext))
    if a != b:
        raise SystemExit(f"{images} was preprocessed with {b}, but {run} trained on {a}. "
                         "Preprocess the external photos the same way.")


def report(df, score, cut_std, cut_caution, n_boot, seed):
    """Print the external results and return them as a dict."""
    sick, sick_dme = df["sick"].to_numpy(), df["sick_or_dme"].to_numpy()
    out = {"n_images": len(df), "n_patients": int(df["patient"].nunique()),
           "n_referable": int(sick.sum()), "n_referable_or_dme": int(sick_dme.sum())}
    boots = list(bootstrap_weights(df["patient"], n_boot, seed))
    print(f"\nEXTERNAL TEST, scored once ({len(df)} gradable photos, {out['n_patients']} patients, "
          f"{out['n_referable']} referable = grade >= 2):")
    a = auc(sick, score)
    out["auc"] = a
    out["auc_95"] = [float(x) for x in np.percentile([auc(sick, score, w) for w in boots], [2.5, 97.5])]
    print(f"  AUC (moderate-or-worse DR)  {a:.3f}   95% range [{out['auc_95'][0]:.3f} to {out['auc_95'][1]:.3f}]")
    for name, cut in (("standard", cut_std), ("cautious", cut_caution)):
        ref = score >= cut
        s, p = sens_spec(sick, ref)
        bs = np.array([sens_spec(sick, ref, w) for w in boots])
        out[name] = {"cutoff": cut, "sensitivity": s, "specificity": p,
                     "sensitivity_95": [float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                     "specificity_95": [float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
                     "share_referred": float(ref.mean())}
        print(f"  {name:8s} cutoff: sensitivity {s:.1%} {range95(bs[:, 0], True)} | "
              f"specificity {p:.1%} {range95(bs[:, 1], True)} | refers {ref.mean():.1%} of photos")
    a2 = auc(sick_dme, score)
    out["auc_referable_or_dme"] = a2
    print(f"  AUC (moderate-or-worse DR OR macular edema, a common published definition): {a2:.3f}")
    print("\nShare referred by true grade (standard / cautious cutoff):")
    for g, part in df.groupby("level"):
        idx = part.index.to_numpy()
        print(f"  grade {g} ({GRADE_NAMES.get(g, '?'):13s}) {np.mean(score[idx] >= cut_std):6.1%} / "
              f"{np.mean(score[idx] >= cut_caution):6.1%} of {len(part):4d} photos")
    per_patient = pd.DataFrame({"patient": df["patient"], "sick": sick, "score": score}).groupby("patient").max()
    ps, pp = sens_spec(per_patient["sick"].to_numpy(), per_patient["score"].to_numpy() >= cut_std)
    out["per_patient_standard"] = {"sensitivity": ps, "specificity": pp}
    print(f"\nPer patient (refer if either eye is flagged, standard cutoff): sensitivity {ps:.1%}, "
          f"specificity {pp:.1%}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tta-runs", nargs="+", required=True,
                   help="the frozen system's <run>_tta folders (two or more = ensemble)")
    p.add_argument("--images", required=True, help="the external photos, preprocessed like the training photos")
    p.add_argument("--grades", required=True, help="grades CSV (Messidor-2 adjudicated format)")
    p.add_argument("--pairs", required=True, help="pairing CSV, one row per patient: left;right")
    p.add_argument("--out", required=True, help="new folder for this external result, e.g. runs/messidor2_v4")
    p.add_argument("--no-both-eyes", action="store_true", help="score each eye alone (default: blend like the system)")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    results_path = os.path.join(args.out, "external_results.json")
    if os.path.exists(results_path):
        raise SystemExit(f"{results_path} exists: this external set was already scored once with this system. "
                         "An external test is looked at once; do not rerun it to get a different number.")

    # 1. The system's choices, all from validation (identical to analyze.py's).
    runs = [source_run(d) for d in args.tta_runs]
    both_eyes = not args.no_both_eyes
    choices = validation_choices(args.tta_runs, len(args.tta_runs) > 1, both_eyes, args.target_sensitivity)
    val = choices["val"]
    cut_std = choices["cut"]
    cut_caution = cautious_cutoff(val["sick"].to_numpy(), choices["combined"], val["patient"],
                                  args.target_sensitivity, args.n_boot, args.seed)
    v_sens, v_spec = sens_spec(val["sick"].to_numpy(), choices["combined"] >= cut_caution)
    print(f"System: {' + '.join(args.tta_runs)} (8-view TTA each)"
          f"{', both eyes w = %.2f' % choices['w'] if both_eyes else ''}")
    print(f"Cutoffs fixed on validation: standard {cut_std:+.4f}; cautious {cut_caution:+.4f} "
          f"(on validation: sensitivity {v_sens:.1%}, specificity {v_spec:.1%})")

    # 2. The external photos, scored once by every model of the system (8 views each).
    df, n_ungradable, n_pairs = load_external(args.grades, args.pairs, args.images)
    print(f"External set: {len(df)} gradable photos of {df['patient'].nunique()} patients "
          f"({n_ungradable} ungradable left out; pairing file has {n_pairs} patients)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    table = df[["image", "patient", "level"]].copy()
    for i, run in enumerate(runs):
        check_recipe(run, args.images)
        ckpt = torch.load(os.path.join(run, "best.pt"), map_location=device)
        arch = ckpt.get("arch", {})
        model = build_model(depth=arch.get("depth", 18), pool=arch.get("pool", "avg"),
                            dropout=arch.get("dropout", 0.2)).to(device)
        model.load_state_dict(ckpt["model"])
        _, eval_tf = make_transforms(ckpt["size"], ckpt["mean"], ckpt["std"])   # the run's own normalization
        loader = DataLoader(FundusDataset(df, args.images, eval_tf), batch_size=args.batch,
                            num_workers=workers, pin_memory=use_amp)
        _, table[f"score_{i}"] = score_eyes(model, loader, device, use_amp)
        print(f"  scored with {run} (epoch {ckpt['epoch']}, device {device})")
    score = np.asarray(apply_choices(choices, table), dtype=np.float64)

    # 3. Results: written right away, so they never have to be produced twice.
    os.makedirs(args.out, exist_ok=True)
    preds = table.assign(dme=df["dme"], system_score=score, referred_standard=score >= cut_std,
                         referred_cautious=score >= cut_caution)
    preds.to_csv(os.path.join(args.out, "external_predictions.csv"), index=False)
    results = report(df, score, cut_std, cut_caution, args.n_boot, args.seed)
    results.update({"system": args.tta_runs, "runs": [os.path.abspath(r) for r in runs], "both_eyes_w": choices["w"],
                    "n_ungradable_excluded": n_ungradable, "grades": os.path.abspath(args.grades),
                    "images": os.path.abspath(args.images)})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {results_path}. This external set is now used: record the numbers, do not rerun.")


if __name__ == "__main__":
    main()
