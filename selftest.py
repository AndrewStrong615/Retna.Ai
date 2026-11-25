"""
One command that checks the whole pipeline on a CPU in about a minute.
Run it after changing any code, before pushing:

    python selftest.py

What it checks:
  1. Threshold picking finds the best possible cutoff (compared with trying
     every cutoff), including when scores tie, as float16 GPU scores often do.
  2. The patient split: no patient in two parts, and the same split every time.
  3. The ordinal loss: cutpoints start at the base rates and always stay in order.
  4. Saving: numpy numbers are converted, so json and torch.load(weights_only=True)
     accept everything we save.
  5. End to end on tiny synthetic data, for both targets (ordinal and binary):
     make data -> preprocess -> train 2 epochs -> test. All output files exist,
     best.pt reloads safely, the saved validation predictions reproduce the logged
     AUC and specificity exactly, and a second run into the same folder is refused.
  6. Stage 2: pretrain.py leaves out validation/test patients; train.py --init loads its encoder.
  7. The switches: preprocess --tight-mask (gray outside the eye, preprocess.json
     written), and a run with --pool gem --depth 34 whose saved architecture
     heatmap.py can rebuild.

It prints PASS or FAIL for each check and exits with an error if anything failed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load_labels, split_by_patient          # noqa: E402
from train import (CumulativeLogitLoss, no_skill_loss, pick_threshold,  # noqa: E402
                   safe_auc, sens_spec, to_py)

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def run(args, cwd=HERE):
    """Run one of the project's scripts with this Python; return (exit code, output)."""
    out = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def test_threshold():
    rng = np.random.default_rng(0)
    bad = 0
    for case in range(150):
        n = int(rng.integers(40, 800))
        labels = rng.random(n) < 0.25
        if labels.all() or not labels.any():
            continue
        scores = rng.normal(size=n) + 1.2 * labels
        if case % 2:                                   # half the cases: float16-style ties
            scores = scores.astype(np.float16).astype(np.float32)
        thr = pick_threshold(labels, scores, 0.9)
        sens, spec = sens_spec(labels, scores, thr)
        best = max(sens_spec(labels, scores, c)[1] for c in np.unique(scores)
                   if sens_spec(labels, scores, c)[0] >= 0.9)
        bad += sens < 0.9 or spec < best - 1e-12
    check("threshold picking is optimal (150 random cases, half with ties)", bad == 0,
          f"{bad} suboptimal")


def test_split(labels_csv):
    df = load_labels(labels_csv)
    a = split_by_patient(df, seed=42)
    b = split_by_patient(df, seed=42)
    same = all(x["image"].tolist() == y["image"].tolist() for x, y in zip(a, b))
    patients = [set(part["patient"]) for part in a]
    disjoint = not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2])
    check("patient split is repeatable and keeps both eyes of a patient together", same and disjoint)


def test_ordinal_loss():
    grades = np.array([0] * 70 + [1] * 8 + [2] * 15 + [3] * 4 + [4] * 3)
    loss = CumulativeLogitLoss(grades)
    above = np.array([(grades > k).mean() for k in range(4)])
    start_ok = np.allclose(loss.cutpoints().detach().numpy(), np.log(above / (1 - above)), atol=1e-4)
    with torch.no_grad():                              # push the gaps toward zero and negative
        loss.raw_gaps.copy_(torch.tensor([-30.0, 5.0, -30.0]))
    ordered = bool((loss.cutpoints().diff() < 0).all())
    fresh = CumulativeLogitLoss(grades)
    with torch.no_grad():
        at_zero = fresh(torch.zeros(len(grades)), torch.from_numpy(grades)).item()
    no_skill = no_skill_loss(fresh, grades, torch.device("cpu"))
    check("ordinal loss: cutpoints start at the base rates", start_ok)
    check("ordinal loss: cutpoints stay in order", ordered)
    check("ordinal loss: an untrained model sits at the no-skill loss", abs(at_zero - no_skill) < 1e-3,
          f"{at_zero:.4f} vs {no_skill:.4f}")


def test_saving(tmp):
    obj = to_py({"a": np.float64(0.5), "b": np.int64(3), "c": np.bool_(True),
                 "d": np.arange(3), "e": torch.tensor(2.5), 4: (np.float32(1.5),)})
    json.dumps(obj)
    path = os.path.join(tmp, "x.pt")
    torch.save({"w": torch.zeros(2), **obj}, path)
    loaded = torch.load(path, weights_only=True)
    check("saving: json and torch.load(weights_only=True) accept converted values",
          loaded["b"] == 3 and loaded["d"] == [0, 1, 2])


def test_end_to_end(tmp):
    data = os.path.join(tmp, "synthetic")
    code, log = run(["make_synthetic_data.py", "--out", data, "--patients", "60"])
    check("make synthetic data", code == 0, log.strip().splitlines()[-1] if log else "")
    processed = os.path.join(data, "processed_96")
    code, log = run(["preprocess.py", "--src", os.path.join(data, "raw"), "--dst", processed,
                     "--size", "96", "--workers", "2"])
    check("preprocess (the one corrupt file is expected to fail)", code == 0 and "1 images failed" in log)
    labels = os.path.join(data, "trainLabels.csv")
    test_split(labels)

    for target in ("ordinal", "binary"):
        out = os.path.join(tmp, f"run_{target}")
        args = ["train.py", "--labels", labels, "--images", processed, "--size", "96",
                "--epochs", "2", "--batch", "8", "--workers", "0", "--target", target, "--out", out]
        code, log = run(args)
        if code != 0:
            print(log[-3000:])
        check(f"[{target}] training runs to the end", code == 0 and "TEST (best epoch" in log)
        if code != 0:
            continue
        expected = ["config.json", "history.json", "best.pt", "val_predictions.csv",
                    "test_predictions.csv", "test_results.json"]
        missing = [f for f in expected if not os.path.exists(os.path.join(out, f))]
        check(f"[{target}] all output files written", not missing, f"missing {missing}" if missing else "")

        ckpt = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=True)
        check(f"[{target}] best.pt reloads with weights_only=True", "loss" in ckpt and ckpt["target"] == target)

        val = pd.read_csv(os.path.join(out, "val_predictions.csv"), converters={"score": float})
        labels_v = val["level"] >= 2
        history = json.load(open(os.path.join(out, "history.json")))
        best_row = next(r for r in history if r["epoch"] == ckpt["epoch"])
        sens_v, spec_v = sens_spec(labels_v, val["score"], ckpt["threshold"])
        check(f"[{target}] saved val predictions reproduce the logged AUC and specificity",
              safe_auc(labels_v, val["score"]) == ckpt["val_auc"] == best_row["val_auc"]
              and abs(spec_v - best_row["val_specificity"]) < 1e-12)
        check(f"[{target}] the `referred` column matches the stored threshold (val)",
              (val["referred"] == (val["score"] >= ckpt["threshold"])).all() and sens_v >= 0.9)

        results = json.load(open(os.path.join(out, "test_results.json")))
        check(f"[{target}] test_results.json is complete",
              {"auc", "sensitivity", "specificity", "threshold", "best_epoch"} <= set(results))
        test = pd.read_csv(os.path.join(out, "test_predictions.csv"), converters={"score": float})
        labels_t = test["level"] >= 2
        sens_t, spec_t = sens_spec(labels_t, test["referred"], True)
        check(f"[{target}] test predictions reproduce test_results.json",
              abs(safe_auc(labels_t, test["score"]) - results["auc"]) < 1e-12
              and (sens_t, spec_t) == (results["sensitivity"], results["specificity"])
              and sens_spec(labels_t, test["score"], results["threshold"]) == (sens_t, spec_t))

        code, log = run(args)
        check(f"[{target}] a second run into the same folder is refused",
              code != 0 and "already holds results" in log)

    # Predicted 0-4 grade from the ordinal cutpoints: saved by train.py, reproduced by grade_report.py,
    # hand-written kappa equal to scikit-learn's, grade probabilities sum to 1; binary runs refused.
    from sklearn.metrics import cohen_kappa_score
    from grade_report import load_cutpoints, weighted_kappa
    run_o_dir = os.path.join(tmp, "run_ordinal")
    vo = pd.read_csv(os.path.join(run_o_dir, "val_predictions.csv"), converters={"score": float})
    lossf, _ = load_cutpoints(run_o_dir)
    with torch.no_grad():
        again = lossf.predicted_grade(torch.tensor(vo["score"].to_numpy())).numpy()
        probs = lossf.grade_probs(torch.linspace(-8, 8, 50))
        median = (probs.cumsum(dim=1) < 0.5).sum(dim=1)      # first grade where the cumulative share reaches 0.5
        direct = lossf.predicted_grade(torch.linspace(-8, 8, 50))
    rng = np.random.default_rng(0)
    t_k = rng.integers(0, 5, 300)
    p_k = np.clip(t_k + rng.integers(-2, 3, 300), 0, 4)
    code_g, log_g = run(["grade_report.py", run_o_dir, "--n-boot", "50"])
    code_gb, log_gb = run(["grade_report.py", os.path.join(tmp, "run_binary")])
    check("predicted grade: saved = recomputed, median of the grade probabilities, kappa = sklearn's, binary refused",
          (vo["predicted_grade"].to_numpy() == again).all() and torch.allclose(probs.sum(dim=1), torch.ones(50))
          and (probs >= 0).all() and torch.equal(median, direct)
          and abs(weighted_kappa(t_k, p_k) - cohen_kappa_score(t_k, p_k, weights="quadratic")) < 1e-12
          and code_g == 0 and "weighted kappa" in log_g and code_gb != 0 and "only the ordinal" in log_gb,
          (log_g or log_gb).strip().splitlines()[-1] if (log_g or log_gb) else "")

    # analyze.py on the two runs above: a single-run report and a paired comparison.
    run_o, run_b = os.path.join(tmp, "run_ordinal"), os.path.join(tmp, "run_binary")
    code1, log1 = run(["analyze.py", run_o, "--n-boot", "50"])
    code2, log2 = run(["analyze.py", run_o, run_b, "--n-boot", "50"])
    code3, log3 = run(["analyze.py", run_o, run_b, "--split", "test"])
    check("analyze.py: report, paired comparison, and refusal to compare on test",
          code1 == 0 and "Per patient" in log1 and code2 == 0 and "Verdict" in log2
          and code3 != 0 and "validation only" in log3)
    code4, log4 = run(["analyze.py", run_o, run_b, "--ensemble", "--both-eyes", "--final-test", "--n-boot", "50"])
    check("analyze.py: ensemble + both eyes, chosen on validation, applied to test once",
          code4 == 0 and "chosen w =" in log4 and "TEST SET, evaluated once" in log4)

    # TTA: the 8 orientations are 8 different exact pixel shuffles; view 1 reproduces the run;
    # validation must come first and the test eyes can be scored only once.
    sys.path.insert(0, HERE)
    from tta import eight_views
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    views = eight_views(x)
    check("tta: 8 distinct orientations that only move pixels",
          len({tuple(v.flatten().tolist()) for v in views}) == 8
          and all(torch.equal(v.flatten().sort().values, x.flatten().sort().values) for v in views))
    tta_out = os.path.join(tmp, "run_ordinal_tta")
    tta_args = ["tta.py", "--run", run_o, "--images", processed, "--workers", "0", "--out", tta_out]
    code_t0, log_t0 = run(tta_args + ["--split", "test"])
    code_v, log_v = run(tta_args)
    plain_ok = False
    if code_v == 0:
        info = json.load(open(os.path.join(tta_out, "tta.json")))
        ckpt_o = torch.load(os.path.join(run_o, "best.pt"), map_location="cpu", weights_only=True)
        plain_ok = abs(info["val_auc_plain"] - ckpt_o["val_auc"]) < 1e-9
    code_t, log_t = run(tta_args + ["--split", "test"])
    code_t2, log_t2 = run(tta_args + ["--split", "test"])
    code_v2, log_v2 = run(tta_args)
    code_a, log_a = run(["analyze.py", tta_out, "--both-eyes", "--final-test", "--n-boot", "50"])
    check("tta: view 1 reproduces the run's validation AUC", plain_ok)
    check("tta: validation first, test once, then validation is frozen",
          code_t0 != 0 and code_v == 0 and code_t == 0 and code_t2 != 0 and code_v2 != 0)
    check("tta: analyze.py reads its output (both eyes + final test)",
          code_a == 0 and "TEST SET, evaluated once" in log_a)

    # tta.py --eval-size: scores at another size into its own folder, records the size, and the
    # test pass refuses a size other than the one the validation cutoff was chosen at.
    tta_s = os.path.join(tmp, "run_ordinal_tta_s112")
    s_args = ["tta.py", "--run", run_o, "--images", processed, "--workers", "0", "--out", tta_s]
    code_s, log_s = run(s_args + ["--eval-size", "112"])
    size_ok = code_s == 0 and json.load(open(os.path.join(tta_s, "tta.json")))["eval_size"] == 112
    code_s2, _ = run(s_args + ["--split", "test", "--eval-size", "96"])
    code_s3, _ = run(s_args + ["--split", "test"])
    check("tta --eval-size: own size recorded, test pass must reuse it",
          size_ok and code_s2 != 0 and code_s3 == 0, log_s.strip().splitlines()[-1] if log_s else "")

    # --ema: the average follows d * avg + (1 - d) * live (d growing from small at the start),
    # and a run with it saves the averaged network plus the live network's best epoch.
    from train import WeightAverage
    live = torch.nn.Linear(2, 1)
    avg = WeightAverage([live], 0.5)
    start = live.weight.detach().clone()
    with torch.no_grad():
        live.weight.add_(1.0)
    avg.update([live])                                 # n = 1: d = min(0.5, 2/11) = 2/11
    want = start * 2 / 11 + live.weight.detach() * 9 / 11
    check("ema: one update gives d * average + (1 - d) * live, d = min(decay, (1+n)/(10+n))",
          torch.allclose(avg.copies[0].weight, want))
    out = os.path.join(tmp, "run_ema")
    code, log = run(["train.py", "--labels", labels, "--images", processed, "--size", "96", "--epochs", "2",
                     "--batch", "8", "--workers", "0", "--ema", "0.99", "--out", out])
    code_a, log_a = run(["analyze.py", os.path.join(out, "live"), out, "--n-boot", "50"]) if code == 0 else (1, "")
    ema_ok = code == 0 and torch.load(os.path.join(out, "best.pt"), weights_only=True)["ema"] == 0.99
    check("train.py --ema runs, saves the average, and analyze.py pairs it with the live weights",
          ema_ok and code_a == 0 and "val_auc_live" in json.load(open(os.path.join(out, "history.json")))[0],
          log.strip().splitlines()[-1] if log else "")

    # --aug strong (exp17): the camera steps give valid, different pictures; a short run trains with it.
    from data import make_transforms
    from PIL import Image as PILImage
    photo = PILImage.open(os.path.join(processed, sorted(f for f in os.listdir(processed) if f.endswith(".jpg"))[0]))
    std_tf, _ = make_transforms(96, [0.5] * 3, [0.25] * 3, (128, 128, 128), aug="standard")
    strong_tf, _ = make_transforms(96, [0.5] * 3, [0.25] * 3, (128, 128, 128), aug="strong")
    torch.manual_seed(0)
    views = [strong_tf(photo) for _ in range(20)]
    check("aug strong: right shape, finite values, every view different",
          all(v.shape == std_tf(photo).shape and torch.isfinite(v).all() for v in views)
          and len({round(float(v.sum()), 3) for v in views}) == 20)
    out = os.path.join(tmp, "run_aug")
    code, log = run(["train.py", "--labels", labels, "--images", processed, "--size", "96", "--epochs", "2",
                     "--batch", "8", "--workers", "0", "--aug", "strong", "--out", out])
    check("train.py --aug strong runs and records it in config.json",
          code == 0 and json.load(open(os.path.join(out, "config.json"))).get("aug") == "strong",
          log.strip().splitlines()[-1] if log else "")

    # Stage 2: pretraining leaves out validation/test patients, and train.py --init uses its encoder.
    pre = os.path.join(tmp, "pre")
    code, log = run(["pretrain.py", "--images", processed, "--exclude-labels", labels, "--size", "96",
                     "--epochs", "1", "--batch", "8", "--workers", "0", "--out", pre])
    check("pretrain.py runs and writes encoder.pt", code == 0 and os.path.exists(os.path.join(pre, "encoder.pt")))
    if code == 0:
        sys.path.insert(0, HERE)
        from pretrain import list_images
        _, val_p, test_p = split_by_patient(load_labels(labels), seed=42)
        paths, _, _ = list_images([processed], labels, 42)
        used = {os.path.basename(x)[:-4].split("_")[0] for x in paths}
        check("pretraining never sees validation or test patients",
              not (used & (set(val_p["patient"]) | set(test_p["patient"]))))
        out = os.path.join(tmp, "run_init")
        code, log = run(["train.py", "--labels", labels, "--images", processed, "--size", "96", "--epochs", "1",
                         "--batch", "8", "--workers", "0", "--pool", "gem", "--init",
                         os.path.join(pre, "encoder.pt"), "--out", out])
        loaded = False
        if code == 0:
            enc = torch.load(os.path.join(pre, "encoder.pt"), weights_only=True)["encoder"]
            from model import build_model
            m = build_model(18, "gem")
            m.load_state_dict(enc, strict=False)
            loaded = json.load(open(os.path.join(out, "config.json")))["init"] is not None
        check("train.py --init starts from the pretrained encoder and records it", code == 0 and loaded)

    # Gradient accumulation: --accum 2 runs (pieces of 4 images) and is recorded.
    out = os.path.join(tmp, "run_accum")
    code, log = run(["train.py", "--labels", labels, "--images", processed, "--size", "96", "--epochs", "1",
                     "--batch", "8", "--accum", "2", "--workers", "0", "--out", out])
    check("train.py --accum 2 runs and records it",
          code == 0 and json.load(open(os.path.join(out, "config.json")))["accum"] == 2)

    # Extra labeled photos (training only) and the labels authenticity check. A second synthetic
    # set gets new patient numbers (+100000), like the EyePACS test folder's own patients.
    data2 = os.path.join(tmp, "synthetic2")
    run(["make_synthetic_data.py", "--out", data2, "--patients", "30", "--seed", "1"])
    p2 = os.path.join(data2, "processed_96")
    run(["preprocess.py", "--src", os.path.join(data2, "raw"), "--dst", p2, "--size", "96", "--workers", "2"])
    renamed = os.path.join(data2, "renamed")
    os.makedirs(renamed)
    shutil.copy(os.path.join(p2, "preprocess.json"), renamed)
    new_name = lambda n: f"{int(n.split('_')[0]) + 100000}_{n.split('_')[1]}"
    lab2 = pd.read_csv(os.path.join(data2, "trainLabels.csv"))
    for n in lab2["image"]:
        if os.path.exists(os.path.join(p2, f"{n}.jpg")):
            shutil.copy(os.path.join(p2, f"{n}.jpg"), os.path.join(renamed, f"{new_name(n)}.jpg"))
    extra_csv = os.path.join(data2, "extra.csv")
    lab2.assign(image=lab2["image"].map(new_name)).to_csv(extra_csv, index=False)
    _, val_p, _ = split_by_patient(load_labels(labels), seed=42)
    bad_csv = os.path.join(data2, "extra_bad.csv")
    pd.concat([pd.read_csv(extra_csv), val_p[["image", "level"]].head(1)]).to_csv(bad_csv, index=False)
    base = ["train.py", "--labels", labels, "--images", processed, "--size", "96", "--epochs", "1",
            "--batch", "8", "--workers", "0"]
    out = os.path.join(tmp, "run_extra")
    # 2 epochs here: check_labels.py below needs a model that has learned something.
    code, log = run([a if a != "1" else "2" for a in base] + ["--extra-labels", extra_csv,
                                                              "--extra-images", renamed, "--out", out])
    same_val = code == 0 and pd.read_csv(os.path.join(out, "val_predictions.csv"))["image"].tolist() == \
        pd.read_csv(os.path.join(run_o, "val_predictions.csv"))["image"].tolist()
    extra_rec = json.load(open(os.path.join(out, "config.json")))["extra"] if code == 0 else None
    check("train.py --extra-labels: extra photos join training only, validation eyes unchanged",
          same_val and "extra:" in log and extra_rec and extra_rec[0]["n_images"] > 50)
    code_b, log_b = run(base + ["--extra-labels", bad_csv, "--extra-images", renamed,
                                "--out", os.path.join(tmp, "run_extra_bad")])
    with open(os.path.join(renamed, "preprocess.json")) as f:
        recipe = json.load(f)
    with open(os.path.join(renamed, "preprocess.json"), "w") as f:
        json.dump(dict(recipe, size=128), f)
    code_m, log_m = run(base + ["--extra-labels", extra_csv, "--extra-images", renamed,
                                "--out", os.path.join(tmp, "run_extra_mismatch")])
    with open(os.path.join(renamed, "preprocess.json"), "w") as f:
        json.dump(recipe, f)
    check("train.py --extra-labels refuses validation/test patients and differently preprocessed photos",
          code_b != 0 and "validation/test patients" in log_b and code_m != 0 and "preprocessed differently" in log_m)
    shuffled_csv = os.path.join(data2, "extra_shuffled.csv")
    sh = pd.read_csv(extra_csv)
    sh.assign(level=sh["level"].sample(frac=1, random_state=0).to_numpy()).to_csv(shuffled_csv, index=False)
    chk = ["check_labels.py", "--run", out, "--images", renamed, "--workers", "0", "--labels"]
    code_g, log_g = run(chk + [extra_csv])
    code_s, log_s = run(chk + [shuffled_csv])
    code_c, log_c = run(chk + [bad_csv])
    auc_of = lambda log: float(log.split("AUC against the file's grades:")[1].split()[0])
    check("check_labels.py: real grades score higher than shuffled ones; held-out patients refused",
          code_g == 0 and code_s == 0 and auc_of(log_g) > auc_of(log_s)
          and code_c != 0 and "validation/test patients" in log_c)

    # DDR: the converter (drops ungradable grade 5 and label lines without a photo), preprocess.py
    # with several source folders (and its refusal of duplicate names), and training with TWO
    # extra datasets at once. A fake DDR layout: DR_grading/{train,valid,test}.txt + photo folders.
    ddr = os.path.join(data2, "DR_grading")
    # Only photos that preprocess fine (the synthetic set has one deliberately corrupt file).
    raw2 = sorted(f for f in os.listdir(os.path.join(data2, "raw"))
                  if f.endswith(".jpeg") and os.path.exists(os.path.join(p2, f[:-5] + ".jpg")))
    lab2_map = dict(zip(lab2["image"], lab2["level"]))
    expected = 0
    for k, part in enumerate(("train", "valid", "test")):
        os.makedirs(os.path.join(ddr, part))
        lines = []
        for j, f in enumerate(raw2[k * 12:(k + 1) * 12]):
            name = f"{k:03d}-{j:04d}-000.jpg"            # DDR-style name: no "_", no patient number
            grade = 5 if j == 0 else lab2_map[f[:-5]]    # one ungradable photo per part
            if j != 1:                                   # one label line without its photo per part
                shutil.copy(os.path.join(data2, "raw", f), os.path.join(ddr, part, name))
            lines.append(f"{name} {grade}")
            expected += j not in (0, 1)
        with open(os.path.join(ddr, f"{part}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    ddr_csv = os.path.join(data2, "ddr_labels.csv")
    code_c, log_c = run(["convert_ddr.py", "--root", ddr, "--out", ddr_csv])
    rows = pd.read_csv(ddr_csv) if code_c == 0 else pd.DataFrame()
    check("convert_ddr.py: keeps gradable photos with a file, drops grade 5 and missing photos",
          code_c == 0 and len(rows) == expected and rows["level"].between(0, 4).all())
    ddr_proc = os.path.join(data2, "processed_ddr")
    code_p, log_p = run(["preprocess.py", "--src"] + [os.path.join(ddr, q) for q in ("train", "valid", "test")]
                        + ["--dst", ddr_proc, "--size", "96", "--workers", "2"])
    n_out = len([f for f in os.listdir(ddr_proc) if f.endswith(".jpg")]) if code_p == 0 else 0
    dup = os.path.join(data2, "dup_src")
    os.makedirs(dup)
    shutil.copy(os.path.join(ddr, "train", "000-0002-000.jpg"), dup)
    code_d, log_d = run(["preprocess.py", "--src", os.path.join(ddr, "train"), dup,
                         "--dst", os.path.join(data2, "processed_dup"), "--size", "96", "--workers", "2"])
    check("preprocess.py: several source folders into one, refuses duplicate names",
          code_p == 0 and n_out == 33 and code_d != 0 and "names must be unique" in log_d)
    out = os.path.join(tmp, "run_two_extras")
    code, log = run(base + ["--extra-labels", extra_csv, ddr_csv, "--extra-images", renamed, ddr_proc, "--out", out])
    rec = json.load(open(os.path.join(out, "config.json")))["extra"] if code == 0 else []
    check("train.py: two extra datasets at once, each recorded",
          code == 0 and len(rec) == 2 and rec[1]["n_images"] == expected and "train now" in log)

    # --source-mix: the sampler keeps every batch's mix within one photo of the shares, uses every photo of a
    # source before repeating one, and train.py records it; a wrong number of shares is refused.
    from data import SourceMixSampler
    src = np.array([0] * 100 + [1] * 300 + [2] * 40)
    smp = SourceMixSampler(src, [2, 2, 1], seed=0)
    epochs = [np.array(list(smp)) for _ in range(3)]
    quota = np.array([0.4, 0.4, 0.2]) * 32
    batch_ok = all(np.all(np.abs(np.bincount(src[e[i:i + 32]], minlength=3) - quota) <= 1)
                   for e in epochs for i in range(0, len(e) - 31, 32))
    seen = np.concatenate(epochs)
    # source 1 (300 photos, 176 slots/epoch): 528 draws over 3 epochs = every photo once, then 228 again
    reps = np.bincount(seen[src[seen] == 1], minlength=400)[100:400]
    small = np.bincount(seen[src[seen] == 2], minlength=440)[400:]      # 40 photos, 264 draws: 6-7 each
    out_mix = os.path.join(tmp, "run_mix")
    code_m, log_m = run(base + ["--extra-labels", extra_csv, ddr_csv, "--extra-images", renamed, ddr_proc,
                                "--source-mix", "0.4", "0.4", "0.2", "--out", out_mix])
    mix_rec = json.load(open(os.path.join(out_mix, "config.json")))["source_mix"] if code_m == 0 else None
    code_m2, log_m2 = run(base + ["--extra-labels", extra_csv, "--extra-images", renamed,
                                  "--source-mix", "0.4", "0.4", "0.2", "--out", os.path.join(tmp, "run_mix_bad")])
    check("train.py --source-mix: batches hold the mix, photos used evenly, recorded; wrong count refused",
          batch_ok and len(seen) == 3 * 440 and set(reps) <= {1, 2} and set(small) <= {6, 7}
          and code_m == 0 and mix_rec and np.allclose(mix_rec["shares"], [0.4, 0.4, 0.2])
          and "source mix per batch" in log_m and code_m2 != 0 and "needs 2 shares" in log_m2)

    # External test (Stage 4), on a synthetic "other hospital" in Messidor-2's file formats:
    # grades with image_id/adjudicated_* columns, a left;right pairing file, one ungradable photo.
    run_b_dir = os.path.join(tmp, "run_binary")
    tta_b = os.path.join(tmp, "run_binary_tta")
    run(["tta.py", "--run", run_b_dir, "--images", processed, "--workers", "0", "--out", tta_b])
    ext = pd.read_csv(extra_csv)
    ext["patient"] = ext["image"].str.split("_").str[0]
    ext["side"] = ext["image"].str.split("_").str[1]
    has_img = ext["image"].map(lambda n: os.path.exists(os.path.join(renamed, f"{n}.jpg")))
    # Like the real Messidor-2 file: an ungradable photo has an EMPTY grade (and DME).
    pd.DataFrame({"image_id": ext["image"] + ".png",
                  "adjudicated_dr_grade": ext["level"].where(has_img),
                  "adjudicated_dme": pd.Series(0, index=ext.index).where(has_img),
                  "adjudicated_gradable": has_img.astype(int)}).to_csv(
        os.path.join(data2, "ext_grades.csv"), index=False)
    pairs = ext.pivot(index="patient", columns="side", values="image")
    with open(os.path.join(data2, "ext_pairs.csv"), "w") as f:      # same layout as messidor-2.csv
        f.write("left;right \n" + "".join(f"{l}.png;{r}.png\n" for l, r in zip(pairs["left"], pairs["right"])))
    ext_args = ["external_test.py", "--tta-runs", tta_out, tta_b, "--images", renamed,
                "--grades", os.path.join(data2, "ext_grades.csv"), "--pairs", os.path.join(data2, "ext_pairs.csv"),
                "--n-boot", "50", "--workers", "0"]
    ext_out = os.path.join(tmp, "external")
    code_e, log_e = run(ext_args + ["--out", ext_out])
    if code_e != 0:
        print(log_e[-3000:])
    res = json.load(open(os.path.join(ext_out, "external_results.json"))) if code_e == 0 else {}
    check("external_test.py: scores once with validation choices; cautious cutoff refers at least as many",
          code_e == 0 and "EXTERNAL TEST" in log_e and res.get("n_patients") == len(pairs)
          and res["cautious"]["cutoff"] <= res["standard"]["cutoff"]
          and res["cautious"]["share_referred"] >= res["standard"]["share_referred"]
          and res["n_ungradable_excluded"] == int((~has_img).sum()))
    code_e2, log_e2 = run(ext_args + ["--out", ext_out])
    with open(os.path.join(renamed, "preprocess.json")) as f:
        recipe = json.load(f)
    with open(os.path.join(renamed, "preprocess.json"), "w") as f:
        json.dump(dict(recipe, size=128), f)
    code_e3, log_e3 = run(ext_args + ["--out", os.path.join(tmp, "external_mismatch")])
    with open(os.path.join(renamed, "preprocess.json"), "w") as f:
        json.dump(recipe, f)
    check("external_test.py refuses a second scoring and differently preprocessed photos",
          code_e2 != 0 and "already scored once" in log_e2 and code_e3 != 0 and "preprocessed with" in log_e3)

    # APTOS format (id_code, diagnosis; no pairing file): every photo is its own patient, both-eyes off.
    aptos_csv = os.path.join(data2, "aptos_grades.csv")
    ext[has_img].assign(id_code=ext["image"], diagnosis=ext["level"])[["id_code", "diagnosis"]].to_csv(
        aptos_csv, index=False)
    code_a, log_a = run(["external_test.py", "--tta-runs", tta_out, "--images", renamed, "--grades", aptos_csv,
                         "--n-boot", "50", "--workers", "0", "--out", os.path.join(tmp, "external_aptos")])
    res_a = json.load(open(os.path.join(tmp, "external_aptos", "external_results.json"))) if code_a == 0 else {}
    check("external_test.py reads APTOS grades without pairing: one photo = one patient, no both-eyes",
          code_a == 0 and "APTOS grades" in log_a and res_a.get("n_patients") == res_a.get("n_images") == int(has_img.sum())
          and res_a.get("both_eyes_w") == 0 and "macular edema" not in log_a)

    # compare_external.py: a second system on the SAME photos, compared paired; it must reproduce each
    # folder's own numbers (same --n-boot and seed) and refuse two folders that scored different photos.
    aptos_b = os.path.join(tmp, "external_aptos_b")
    run(["external_test.py", "--tta-runs", tta_b, "--images", renamed, "--grades", aptos_csv,
         "--n-boot", "50", "--workers", "0", "--out", aptos_b])
    code_c, log_c = run(["compare_external.py", os.path.join(tmp, "external_aptos"), aptos_b, "--n-boot", "50"])
    code_c2, log_c2 = run(["compare_external.py", ext_out, aptos_b, "--n-boot", "50"])
    check("compare_external.py: paired, reproduces each system's own results, refuses different photo sets",
          code_c == 0 and log_c.count("matches external_results.json") == 2 and "B - A" in log_c
          and code_c2 != 0 and "did not score the same photos" in log_c2,
          (log_c or log_c2).strip().splitlines()[-1] if (log_c or log_c2) else "")

    # AdaBN: BatchNorm statistics become the plain average of the calibration batches' statistics,
    # while every weight stays as it was and the model ends in evaluation mode.
    from external_test import adapt_batchnorm, two_halves
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                              torch.nn.Dropout(0.5))
    weights = net[0].weight.detach().clone()
    batches = [(torch.randn(8, 3, 10, 10) * 2 + 1, None) for _ in range(3)]
    with torch.no_grad():
        want = torch.stack([net[0](x).mean(dim=(0, 2, 3)) for x, _ in batches]).mean(dim=0)
    n_bn = adapt_batchnorm(net, batches, torch.device("cpu"), False)
    halves = two_halves(["a", "a", "b", "c", "c", "d", "e"], 0)
    check("adabn: statistics re-measured as the batch average, weights untouched, eval mode; halves keep patients",
          n_bn == 1 and torch.allclose(net[1].running_mean, want, atol=1e-6) and torch.equal(net[0].weight, weights)
          and not net.training and not net[3].training
          and halves[0] == halves[1] and halves[3] == halves[4] and set(halves) == {0, 1})
    adabn_out = os.path.join(tmp, "external_aptos_adabn")
    code_d, log_d = run(["external_test.py", "--tta-runs", tta_out, "--images", renamed, "--grades", aptos_csv,
                         "--n-boot", "50", "--workers", "0", "--adapt-bn", "--out", adabn_out])
    res_d = json.load(open(os.path.join(adabn_out, "external_results.json"))) if code_d == 0 else {}
    plain_scores = pd.read_csv(os.path.join(tmp, "external_aptos", "external_predictions.csv"))["score_0"]
    adabn_scores = pd.read_csv(os.path.join(adabn_out, "external_predictions.csv"))["score_0"] if code_d == 0 else None
    code_d2, log_d2 = run(["compare_external.py", os.path.join(tmp, "external_aptos"), adabn_out, "--n-boot", "50"])
    check("external_test.py --adapt-bn: both halves scored, recorded, scores change, compare_external pairs it",
          code_d == 0 and log_d.count("AdaBN:") == 2 and res_d.get("adapt_bn") == {"halves": 2, "seed": 0}
          and adabn_scores.notna().all() and not np.allclose(adabn_scores, plain_scores) and code_d2 == 0,
          (log_d or log_d2).strip().splitlines()[-1] if (log_d or log_d2) else "")

    # adapt_site.py: calibrate a "new clinic" on its ungraded photos, then screen. The cutoffs must be the
    # ones external_test.py fixed on validation for the same system (no pairing), and the screening
    # scores must be the adapted models' standardized TTA scores.
    site = os.path.join(tmp, "site_x")
    code_s, log_s = run(["adapt_site.py", "adapt", "--tta-runs", tta_out, "--images", renamed, "--out", site,
                         "--workers", "0", "--n-boot", "50", "--min-photos", "20"])
    code_s2, log_s2 = run(["adapt_site.py", "adapt", "--tta-runs", tta_out, "--images", renamed, "--out", site,
                           "--workers", "0", "--min-photos", "20"])
    code_s3, log_s3 = run(["adapt_site.py", "adapt", "--tta-runs", tta_out, "--images", renamed,
                           "--out", os.path.join(tmp, "site_small"), "--workers", "0"])
    screened = os.path.join(site, "week1.csv")
    code_s4, log_s4 = run(["adapt_site.py", "score", "--site", site, "--images", renamed, "--out", screened,
                           "--workers", "0"])
    site_ok = False
    if code_s == 0 and code_s4 == 0:
        info = json.load(open(os.path.join(site, "site.json")))
        from tta import score_eyes
        from data import FundusDataset, make_transforms
        from model import build_model
        ck = torch.load(os.path.join(site, "model_0.pt"), map_location="cpu")
        net = build_model(**ck["arch"])
        net.load_state_dict(ck["model"])
        orig = torch.load(os.path.join(run_o, "best.pt"), map_location="cpu")
        _, tf = make_transforms(ck["size"], ck["mean"], ck["std"])
        sc = pd.read_csv(screened, dtype={"image": str})
        _, tta_s = score_eyes(net, torch.utils.data.DataLoader(
            FundusDataset(sc.assign(level=0), renamed, tf), batch_size=16), torch.device("cpu"), False)
        mean_, sd_ = info["scales"][0]
        moved = any(not torch.equal(ck["model"][k], orig["model"][k]) for k in ck["model"] if "running_mean" in k)
        same_w = all(torch.equal(ck["model"][k], orig["model"][k]) for k in ck["model"]
                     if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))
        site_ok = (abs(info["cutoff_standard"] - res_a["standard"]["cutoff"]) < 1e-12
                   and abs(info["cutoff_cautious"] - res_a["cautious"]["cutoff"]) < 1e-12
                   and np.allclose(sc["score"], (tta_s - mean_) / sd_, atol=1e-6) and moved and same_w
                   and (sc["referred_standard"] == (sc["score"] >= info["cutoff_standard"])).all()
                   and sc["in_calibration"].all() and info["models"][0]["bn_layers"] == 20)
    check("adapt_site.py: BatchNorm-only calibration, validation cutoffs kept, screening scores reproduce; "
          "refuses a second calibration and too few photos",
          site_ok and code_s2 != 0 and "already calibrated" in log_s2 and code_s3 != 0 and "need at least" in log_s3,
          (log_s or log_s4).strip().splitlines()[-1] if (log_s or log_s4) else "")

    # quality_check.py: fitted on training photos, limits on validation pictures; normal photos from another
    # synthetic "clinic" pass far more often than noise, blank and heavily blurred pictures.
    from PIL import ImageFilter
    odd = os.path.join(tmp, "odd_photos")
    os.makedirs(odd)
    rng_q = np.random.default_rng(0)
    normal = sorted(f for f in os.listdir(renamed) if f.endswith(".jpg"))[:8]
    bg = tuple(int(c) for c in np.asarray(Image.open(os.path.join(renamed, normal[0])).convert("RGB"))[0, 0])
    for i, n in enumerate(normal):
        Image.fromarray(rng_q.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(os.path.join(odd, f"noise_{i}.jpg"))
        Image.new("RGB", (96, 96), bg).save(os.path.join(odd, f"blank_{i}.jpg"))
        Image.open(os.path.join(renamed, n)).filter(ImageFilter.GaussianBlur(4)).save(os.path.join(odd, f"blur_{i}.jpg"))
    code_q, log_q = run(["quality_check.py", "fit", "--run", run_o, "--images", processed, "--quantile", "0.05"])
    q_odd, q_norm = os.path.join(tmp, "q_odd.csv"), os.path.join(tmp, "q_norm.csv")
    code_q2, log_q2 = run(["quality_check.py", "check", "--run", run_o, "--images", odd, "--out", q_odd])
    code_q3, log_q3 = run(["quality_check.py", "check", "--run", run_o, "--images", renamed, "--out", q_norm])
    ok_odd = pd.read_csv(q_odd)["ok"].mean() if code_q2 == 0 else 1
    ok_norm = pd.read_csv(q_norm)["ok"].mean() if code_q3 == 0 else 0
    check("quality_check.py: unusual pictures flagged, normal photos from another clinic mostly pass",
          code_q == 0 and ok_odd <= 0.1 and ok_norm >= 0.7, f"ok: odd {ok_odd:.0%}, normal {ok_norm:.0%}")

    # rescore.py: rebuilds the validation eyes, reproduces the run's own AUC, and also reads a pre-revamp
    # style checkpoint (no arch/target/val_auc; AUC from history.json); analyze.py pairs the result.
    code_r, log_r = run(["rescore.py", "--run", run_o, "--labels", labels, "--images", processed, "--workers", "0"])
    resc = os.path.join(tmp, "run_ordinal_rescored")
    rv = pd.read_csv(os.path.join(resc, "val_predictions.csv"), converters={"score": float}) if code_r == 0 else None
    ov = pd.read_csv(os.path.join(run_o, "val_predictions.csv"), converters={"score": float})
    old_run = os.path.join(tmp, "run_old_style")
    os.makedirs(old_run)
    full = torch.load(os.path.join(run_o, "best.pt"), map_location="cpu")
    torch.save({k: full[k] for k in ("model", "mean", "std", "size", "threshold", "epoch")},
               os.path.join(old_run, "best.pt"))
    shutil.copy(os.path.join(run_o, "history.json"), old_run)
    code_r2, log_r2 = run(["rescore.py", "--run", old_run, "--labels", labels, "--images", processed,
                           "--workers", "0", "--split-after-drop"])
    code_r3, log_r3 = run(["analyze.py", resc, run_b, "--n-boot", "50"])
    code_r4, log_r4 = run(["rescore.py", "--run", run_o, "--labels", labels, "--images", processed, "--workers", "0"])
    check("rescore.py: same validation eyes and AUC as the run, old-style checkpoint read, analyze pairs it, once only",
          code_r == 0 and rv["image"].tolist() == ov["image"].tolist() and np.allclose(rv["score"], ov["score"], atol=1e-6)
          and "-> matches" in log_r and code_r2 == 0 and "-> matches" in log_r2 and code_r3 == 0 and "Verdict" in log_r3
          and code_r4 != 0 and "already rescored" in log_r4,
          (log_r or log_r2).strip().splitlines()[-1] if (log_r or log_r2) else "")

    # 6. The switches.
    tight = os.path.join(data, "processed_96_tight")
    code, log = run(["preprocess.py", "--src", os.path.join(data, "raw"), "--dst", tight,
                     "--size", "96", "--enhance", "--tight-mask", "--workers", "2"])
    info = json.load(open(os.path.join(tight, "preprocess.json"))) if code == 0 else {}
    sample = sorted(f for f in os.listdir(tight) if f.endswith(".jpg"))[0] if code == 0 else None
    corner = np.asarray(Image.open(os.path.join(tight, sample)).convert("RGB"))[0, 0] if sample else [0]
    check("preprocess --tight-mask: gray outside the eye, preprocess.json written",
          code == 0 and info.get("tight_mask") is True and all(abs(int(c) - 128) <= 6 for c in corner))
    out = os.path.join(tmp, "run_gem34")
    code, log = run(["train.py", "--labels", labels, "--images", tight, "--size", "96", "--epochs", "2",
                     "--batch", "8", "--workers", "0", "--pool", "gem", "--depth", "34", "--out", out])
    if code != 0:
        print(log[-3000:])
    arch = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=True).get("arch") \
        if code == 0 else None
    check("train --pool gem --depth 34 runs and saves its architecture",
          arch == {"depth": 34, "pool": "gem", "dropout": 0.2})
    if code == 0:
        code, log = run(["heatmap.py", "--run", out, "--images", tight, "--per-group", "1"])
        pngs = os.listdir(os.path.join(out, "heatmaps")) if code == 0 else []
        check("heatmap.py rebuilds a GeM / depth-34 model and writes pictures", code == 0 and len(pngs) > 0)
        # The 3D eye page: same eyes as heatmap.py, written as one web page plus a table.
        code, log = run(["eye3d.py", "--run", out, "--images", tight, "--per-group", "1"])
        if code != 0:
            print(log[-3000:])
        view = os.path.join(out, "eye3d")
        page = open(os.path.join(view, "viewer.html"), encoding="utf-8").read() if code == 0 else ""
        table = json.load(open(os.path.join(view, "findings.json"))) if code == 0 else {"eyes": []}
        val_names = set(pd.read_csv(os.path.join(out, "val_predictions.csv"))["image"].astype(str))
        check("eye3d.py writes the 3D page (photos embedded, marker filled) for validation eyes only",
              code == 0 and "/*EYE_DATA*/" not in page and page.count("data:image/jpeg;base64,") == len(table["eyes"]) > 0
              and all(e["name"] in val_names for e in table["eyes"]))
        code, log = run(["eye3d.py", "--run", out, "--images", tight, "--image",
                         str(pd.read_csv(os.path.join(out, "test_split.csv"))["image"].iloc[0])])
        check("eye3d.py refuses test images", code != 0 and "TEST images" in log)


def test_eye3d():
    """The hand-written pieces of eye3d.py, on pictures whose right answer is known."""
    from eye3d import MultiScaleCAM, find_findings, label_regions, vessel_sketch
    from heatmap import GradCAM
    from model import build_model
    # Regions: two blobs joined only by a corner count as one; a separate pixel is its own.
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = mask[2:4, 2:4] = mask[5, 5] = True
    labels, count = label_regions(mask)
    check("eye3d: connected regions (corner contact joins, gaps separate)",
          count == 2 and labels[0, 0] == labels[3, 3] != labels[5, 5] and (labels > 0).sum() == mask.sum())
    # Vessel sketch: a long dark line is kept, a round dark blob is not.
    rgb = np.full((128, 128, 3), 150, dtype=np.uint8)
    rgb[62:65, 10:118] = 40                       # the "vessel"
    yy, xx = np.mgrid[:128, :128]
    blob = (yy - 100) ** 2 + (xx - 40) ** 2 < 8 ** 2
    rgb[blob] = 40                                # the "hemorrhage"
    eye = np.ones((128, 128), dtype=bool)
    vessels = vessel_sketch(rgb, eye)
    check("eye3d: vessel sketch keeps a thin dark line and drops a round dark blob",
          vessels[63, 20:108].mean() > 0.9 and vessels[blob].mean() < 0.05)
    # Findings: heat placed on the blob is reported there, as a dark spot, at the right clock position.
    heat = np.exp(-((yy - 100) ** 2 + (xx - 40) ** 2) / (2 * 6.0 ** 2)).astype(np.float32)
    found, numbered = find_findings(heat, rgb, eye, vessels)
    f = found[0] if found else {}
    check("eye3d: a hot dark blob becomes finding #1 (lower left = 7 o'clock, darker, not on a vessel)",
          len(found) == 1 and f["clock"] == 7 and f["tone"] < -0.5 and f["on_vessel_pct"] < 30
          and numbered[100, 40] == 1 and abs(f["x"] - (40 - 63.5) / 64) < 0.03 and abs(f["y"] - (63.5 - 100) / 64) < 0.03)
    # Heat: the region map is exactly heatmap.py's Grad-CAM; the sharpened map stays in 0..1.
    torch.manual_seed(0)
    x = torch.randn(1, 3, 96, 96)
    net_a, net_b = build_model().eval(), build_model().eval()
    for m in net_a.modules():                     # a fresh network's blocks start as pass-throughs with a near-zero
        if isinstance(m, torch.nn.BatchNorm2d):   # head; give it weights that make every layer matter
            torch.nn.init.normal_(m.weight, 1.0, 0.2)
    torch.nn.init.normal_(net_a.head.weight, 0, 1.0)
    net_b.load_state_dict(net_a.state_dict())
    old_map, old_peak, old_score = GradCAM(net_a)(x)
    heat, coarse, peak, score = MultiScaleCAM(net_b, fine_stage=2)(x)
    plain = MultiScaleCAM(build_model().eval(), fine_stage=0)(x)
    check("eye3d: region map = heatmap.py's Grad-CAM; sharpened map in 0..1 with its peak at 1; stage 0 = plain",
          np.allclose(coarse, old_map, atol=1e-6) and abs(peak - old_peak) < 1e-6 and abs(score - old_score) < 1e-6
          and heat.min() >= 0 and abs(heat.max() - 1) < 1e-4 and not np.allclose(heat, coarse)
          and plain[0] is plain[1])


def test_oct3d(tmp):
    """oct3d.py: the surface finder against a volume with a known answer, the readers, and the page."""
    import itertools
    import struct
    from oct3d import best_path, find_layers, load_volume, synthetic_volume, thick_findings, default_threshold
    # Dynamic programming must find the same best path as trying every path (tiny case: 4^5 paths).
    rng = np.random.default_rng(0)
    score = rng.normal(size=(1, 4, 5)).astype(np.float32)
    def total(path):
        moves = [abs(a - b) for a, b in zip(path, path[1:])]
        return -np.inf if max(moves) > 1 else sum(score[0, z, x] for x, z in enumerate(path)) - 0.5 * sum(moves)
    brute = max(itertools.product(range(4), repeat=5), key=total)
    check("oct3d: dynamic programming finds the same best path as trying all 1,024 paths",
          tuple(best_path(score, jump=1, bend_cost=0.5)[0]) == brute)
    vol, spacing, true_ilm, true_low = synthetic_volume()
    ilm, low = find_layers(vol, spacing)
    err_ilm, err_low = np.abs(ilm - true_ilm).mean() * spacing["dz"], np.abs(low - true_low).mean() * spacing["dz"]
    check("oct3d: both surfaces of the synthetic volume found within 8 um on average (pixel = 7.8 um)",
          err_ilm < 8 and err_low < 8, f"ILM {err_ilm:.1f} um, lower {err_low:.1f} um")
    thick = (low - ilm) * spacing["dz"]
    found, _ = thick_findings(thick, spacing, default_threshold(thick))
    f = found[0] if found else {}
    check("oct3d: the one swollen area is found where it was drawn (1.5 mm right, 1 mm toward the first scan, ~500 um)",
          len(found) == 1 and abs(f["x_mm"] - 1.5) < 0.15 and abs(f["y_mm"] + 1.0) < 0.15 and abs(f["peak_um"] - 500) < 25)
    # Readers. Folder: numbers in names order the scans (2 before 10); mixed sizes refused.
    folder = os.path.join(tmp, "oct_folder")
    os.makedirs(folder)
    for i in range(12):
        Image.fromarray(vol[i]).save(os.path.join(folder, f"scan_{i}.png"))
    read, sp, _ = load_volume(folder, 6.0, 1.5, 2.0)
    Image.fromarray(vol[0][:100]).save(os.path.join(folder, "scan_99.png"))
    code, log = run(["oct3d.py", "--scan", folder, "--out", os.path.join(tmp, "oct_bad")])
    check("oct3d: a folder of B-scans is read in numeric order with the given size; mixed sizes refused",
          np.array_equal(read, vol[:12]) and sp["assumed"] and abs(sp["dy"] - 125) < 1e-6
          and code != 0 and "different sizes" in log)
    # Heidelberg .vol: write a small file in the published layout and read it back.
    n, z, x, hdr = 3, 40, 32, 512
    head = bytearray(2048)
    head[:12] = b"HSF-OCT-103\0"
    struct.pack_into("<iiidddii", head, 12, x, n, z, 0.0115, 0.12, 0.0039, 8, 8)
    struct.pack_into("<i", head, 100, hdr)
    data = rng.random((n, z, x)).astype("<f4")
    lines = np.stack([np.full((n, x), 10.0), np.full((n, x), 30.0)], axis=1).astype("<f4")   # device ILM, Bruch's
    blob = bytes(head) + bytes(64)
    for b in range(n):
        bh = bytearray(hdr)
        struct.pack_into("<ii", bh, 48, 2, 64)
        bh[64:64 + 2 * x * 4] = lines[b].tobytes()
        blob += bytes(bh) + data[b].tobytes()
    vol_path = os.path.join(tmp, "eye.vol")
    open(vol_path, "wb").write(blob)
    read, sp, device = load_volume(vol_path)
    check("oct3d: a .vol file in the published layout reads back (sizes, spacing in um, picture, device lines)",
          read.shape == (n, z, x) and abs(sp["dx"] - 11.5) < 1e-3 and abs(sp["dz"] - 3.9) < 1e-3 and not sp["assumed"]
          and np.array_equal(read, (data ** 0.25 * 255).astype(np.uint8))
          and device is not None and np.allclose(device[0], 10) and np.allclose(device[1], 30))
    out = os.path.join(tmp, "oct_demo")
    code, log = run(["oct3d.py", "--demo", "--out", out])
    page = open(os.path.join(out, "viewer.html"), encoding="utf-8").read() if code == 0 else ""
    again, _ = run(["oct3d.py", "--demo", "--out", out])
    check("oct3d.py --demo writes the page, layers.npz and summary.json; refuses to overwrite",
          code == 0 and "/*OCT_DATA*/" not in page and '"ilm_um"' in page
          and os.path.exists(os.path.join(out, "layers.npz"))
          and len(json.load(open(os.path.join(out, "summary.json")))["findings"]) == 1 and again != 0)


def test_vit():
    from vit import _checks
    for name, ok in _checks().items():
        check(f"vit.py (Stage 2b): {name}", ok)


def main():
    test_threshold()
    test_vit()
    test_eye3d()
    test_ordinal_loss()
    with tempfile.TemporaryDirectory() as tmp:
        test_saving(tmp)
        test_oct3d(tmp)
        test_end_to_end(tmp)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
