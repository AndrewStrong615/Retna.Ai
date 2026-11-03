# Retina-from-scratch

A from-scratch deep learning project to detect eye disease from retinal images.
**Stage 1 (current):** classify referable diabetic retinopathy (grade >= 2) from
color fundus photos in the Kaggle EyePACS dataset (`trainLabels.csv`: columns `image,level`).

## Hard rules

- **No pretrained weights, ever.** Every model starts from random initialization.
  No `pretrained=True`, no timm/torchvision/Hugging Face weights, no RETFound,
  no ImageNet checkpoints. PyTorch building blocks (Conv2d, BatchNorm, etc.) are fine.
- **Split by patient, not by image.** Images are named `{patient}_{left|right}`.
  Both eyes of a patient must land in the same split (see `split_by_patient` in `data.py`).
- **The test set is touched once**, at the end of a run. Model selection and the
  decision threshold are chosen on validation data only.
- Never commit dataset images or checkpoints to git. Follow each dataset's license terms.

## Files

- `preprocess.py`: crop black border, pad to square, resize, optional local-average-color enhancement. Run once.
- `model.py`: ResNet-18-style network written from scratch. `python model.py` prints shapes per stage.
- `data.py`: label loading, patient-level split, normalization stats, augmentation, Dataset class.
- `train.py`: training loop (AdamW, warmup + cosine LR, mixed precision), validation AUC and specificity at a
  target sensitivity, best-checkpoint saving, final test evaluation. `--target ordinal` (default) learns the full
  0-4 grade with a cumulative-logit loss; `--target binary` is the original yes/no target with pos_weight.
  `--target binary --rotate-interp nearest --rotate-fill black` reproduces the exp2 recipe exactly (checked
  bit-for-bit against commit 284aabb on CPU). Each run's `--out` folder gets config.json, history.json (per-epoch
  diagnostics), best.pt, val/test_predictions.csv and test_results.json; it refuses a folder that already holds
  results unless `--overwrite`.
- `selftest.py`: one command, ~30 s on CPU: metrics, split, ordinal loss, saving, and a 2-epoch end-to-end run
  for both targets. Run it after every code change, before pushing.
- `make_synthetic_data.py`: tiny fake EyePACS-style dataset (cartoon eyes, obvious lesions) for CPU smoke tests.
  Its docstring has the smoke-test commands.
- `requirements.txt`: dependencies, with the versions the smoke test passed on.
- `kaggle_train.ipynb`: the GPU run on Kaggle (now the backup; the laptop is the main machine): GPU check,
  clone this repo (optionally at `CODE_REF`), preprocess, 2-epoch sanity check, full run, results. Per run, change
  only `EXPERIMENT` (and `TRAIN_ARGS` for options).

## Running an experiment on the laptop (Windows, Command Prompt)

Laptop: Alienware, RTX 5060 Laptop GPU (8 GB), Intel Core 7 240H (16 threads), 32 GB RAM. PyTorch 2.15.0+cu132
in `C:\Users\a1str\Retna.Ai\.venv`. Data in `%USERPROFILE%\retina-data` (from the berinaniesh Kaggle dataset).

    cd %USERPROFILE%\Retna.Ai
    .venv\Scripts\activate
    git pull
    python train.py --labels %USERPROFILE%\retina-data\data\trainLabels.csv --images %USERPROFILE%\retina-data\processed_384_enhanced --out runs\exp3

One-time preprocessing: `python preprocess.py --src %USERPROFILE%\retina-data\data\train --dst %USERPROFILE%\retina-data\processed_384_enhanced --size 384 --enhance`.
Keep the laptop plugged in, awake (no sleep while plugged in) and with heavy apps closed.

## Rules for results

- Choose between runs only with validation numbers (history.json, val_predictions.csv).
- test_results.json and test_predictions.csv are write-only for decisions: recorded in the table, kept for
  Stage 4 and auditing, never used to choose a model, a setting or a threshold.
- Runs with the ordinal target report losses on a different scale; compare runs by AUC (or by "% of no-skill"
  within one target), not by raw loss.

## Status

- Tested on CPU (torch 2.15, Python 3.13): `python model.py` (11.2M parameters, same count as a standard
  ResNet-18 with one output); preprocess -> train -> test end to end on synthetic data, including `--enhance`,
  `--limit`, `--workers 0`, a corrupt image, and a validation set with only one class; patient split
  (no overlap); metric functions (threshold picking matches brute force, also with float16 ties).
  The same seed gives identical results run to run on CPU.
- Smoke test: 3 epochs checks that it runs (~30 s); 20 epochs checks that it learns
  (val AUC 0.68 -> 0.96 on the synthetic data, ~2 min on 4 CPU cores).
- Fixed: when a run never saved a checkpoint, `train.py` could test a stale `best.pt` left in `--out`
  by an earlier run. It now checks whether this run saved one.
- Real training needs an NVIDIA GPU: `kaggle_train.ipynb` runs the pipeline on Kaggle (inputs in `/kaggle/input`,
  outputs to `/kaggle/working`). Use the GPU T4 x2 accelerator: Kaggle's P100 fails with current PyTorch builds
  ("no kernel image is available"). Works on Kaggle, including the GPU path (mixed precision with GradScaler).
  Data: Kaggle dataset `berinaniesh/diabetic-retinopathy-detection` (`data/train`, `data/trainLabels.csv`;
  35,126 photos, all preprocessed without failures). A full run takes ~80 min (~2.4 min/epoch at 384 px on a T4).
- Laptop: the 20-epoch synthetic smoke test passed on the RTX 5060 under Windows (val AUC 0.959, test 0.949),
  including mixed precision and 16 preprocessing workers. Before persistent workers, each epoch paid ~20-30 s of
  Windows worker start-up; train.py now keeps loader workers alive between epochs.
- Revamp, part 1 (after exp2): ordinal target, rotation flags, split made on the full label table before
  dropping missing images (same val/test as exp1/exp2 when nothing is missing), cutoff picked with
  drop_intermediate=False (effect at EyePACS scale well under 0.2 points of sens/spec), threshold stored on the
  raw score, per-epoch diagnostics (no-skill loss, train AUC, lr, img/s, share of time waiting for data, mean val
  score per grade), predictions CSVs, test results written right after the single test pass, numpy values
  converted before saving, refuse-to-overwrite, auto worker count, channels_last + cudnn.benchmark on GPU.
- Planned, part 2: offline `analyze.py` on the predictions CSVs (bootstrap CIs, patient-level and per-grade
  numbers, paired run-vs-run comparison on validation, fellow-eye blending on validation), a val-only re-scorer
  so exp2's best.pt can be paired with later runs, preprocess.json. Later experiments: 60 epochs, 512 px, TTA.

## Experiments

Compare runs by best validation AUC; test numbers are recorded, never used to choose.

| Run | Change from baseline | Best val AUC (epoch) | Test AUC / sens / spec | Notes |
|---|---|---|---|---|
| exp1 | baseline: 384 px, no enhance, 30 epochs, defaults | 0.669 (28) | 0.638 / 87.0% / 23.3% | Underfitting: train loss 1.113 -> 1.037 (no-skill level ~1.11), train ~= val loss, val AUC still rising when the cosine LR reached zero. |
| exp2 | `--enhance` preprocessing (else = exp1) | 0.844 (26) | 0.829 / 88.8% / 48.8% | Starts learning lesions at epoch 3; train loss 1.112 -> 0.768, train ~= val loss (no overfitting); val AUC plateaus ~0.84 from epoch 21 as the LR decays. +0.175 val AUC: enhance is now the baseline. |

Next: exp3 = new defaults (ordinal target, bilinear rotation with background fill) and exp4 = the same with
`--target binary`, back to back on the same split. exp3 vs exp4 is the real test of the ordinal target (it also
drops pos_weight, so call it "ordinal package vs binary"); exp3 vs exp2 is unpaired and changes two things.
Differences under ~0.02-0.025 val AUC are within run-to-run noise until a same-settings seed repeat measures it.

## Working with me

I'm learning ML through this project. Explain what you change and why, keep code well commented,
and don't swap my hand-written components for library versions without asking.

## Roadmap

1. Stage 1: from-scratch CNN for referable DR (this).
2. Stage 2: self-supervised pretraining (masked autoencoder, written from scratch) on pooled unlabeled retinal images.
3. Stage 3: fine-tune that model on multiple diseases (fundus multi-disease, glaucoma, OCT).
4. Stage 4: external validation (e.g. Messidor-2) and comparison with published benchmarks.
