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


def main():
    test_threshold()
    test_ordinal_loss()
    with tempfile.TemporaryDirectory() as tmp:
        test_saving(tmp)
        test_end_to_end(tmp)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
