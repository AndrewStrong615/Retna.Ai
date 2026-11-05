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
  `--tight-mask` (off by default) uses each photo's real eye outline instead of a fixed circle, trims
  `--mask-margin` (2% of the size) at the edge, and with `--enhance` blurs only inside the eye, so no bright rim
  or dark crescents (seen where heat maps drifted on missed eyes). Writes `preprocess.json` into the output folder;
  train.py copies it into each run's config.json.
- `model.py`: ResNet-18-style network written from scratch. `python model.py` prints shapes per stage.
  `build_model(depth, pool, dropout)` builds it: depth 18 (default) or 34 (ResNet-34, 21.3M weights); pool `avg`
  (default) or `gem` (hand-written generalized-mean pooling with learned p, computed in fp32).
- `data.py`: label loading, patient-level split, normalization stats, augmentation, Dataset class.
- `train.py`: training loop (AdamW, warmup + cosine LR, mixed precision), validation AUC and specificity at a
  target sensitivity, best-checkpoint saving, final test evaluation. `--target ordinal` (default) learns the full
  0-4 grade with a cumulative-logit loss; `--target binary` is the original yes/no target with pos_weight.
  `--target binary --rotate-interp nearest --rotate-fill black` is the exp2 recipe (checked bit-for-bit against
  commit 284aabb on CPU with `--workers 0`; with loader workers the augmentation order differs from epoch 2 on,
  so expect run-to-run noise, not identical numbers). Each run's `--out` folder gets config.json, history.json (per-epoch
  diagnostics), best.pt, val/test_predictions.csv and test_results.json; it refuses a folder that already holds
  results unless `--overwrite`. Architecture switches (defaults = exp1-exp6 behavior, checked bit-for-bit):
  `--depth 18|34`, `--pool avg|gem`, `--dropout` (0.2); the architecture is saved in best.pt as `arch`.
- `selftest.py`: one command, ~30 s on CPU: metrics, split, ordinal loss, saving, and a 2-epoch end-to-end run
  for both targets. Run it after every code change, before pushing.
- `bench.py`: speed test that times GPU-only training steps, image loading only (4 and 8 helpers) and both
  together, to find what limits training speed on a machine (~3-4 min; stop training first).
- `heatmap.py`: Grad-CAM written from scratch: for a finished run, colors where the model found evidence of
  disease on validation eyes (missed, false alarms, caught, cleared; refuses test images). Writes
  `<run>/heatmaps/*.png`. Each map is scaled to its own peak; the label's `peak` compares strength across eyes.
- `analyze.py`: offline analysis from the predictions CSVs (no GPU). One run: AUC, sensitivity, specificity with
  95% ranges (bootstrap by patient), share referred per grade, per-patient numbers. Two runs: paired comparison
  on the same validation eyes with a verdict; `--noise-floor` adds measured seed-to-seed noise. Comparisons are
  validation only; `--split test` is for reporting one run's final numbers, never for choosing.
  Free tricks: `--ensemble` (average several runs' standardized scores) and `--both-eyes` (blend each eye's
  score with the fellow eye's, weight w chosen on validation); `--final-test` applies the validation choices
  (scales, w, 90%-sensitivity cutoff) to the test set once, for a frozen result.
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
- Train AUC is measured on augmented images with dropout on, while the weights change, so it normally sits below
  val AUC; overfitting shows as train AUC clearly above val AUC late in a run.
- Predictions CSVs: `referred` is the decision the run made; read `score` with
  `pd.read_csv(path, float_precision="round_trip")` to get it back exactly. For `--target binary` with
  pos_weight, `prob_referable` is inflated by the class weighting (ranking only, not a real risk).

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
  converted before saving, refuse-to-overwrite, auto worker count, cudnn.benchmark on GPU.
- Speed (bench.py on the laptop): GPU-only 301 img/s with channels_last off vs 95 with it on; loading 197 img/s
  with 4 helpers, 363 with 8. channels_last made exp3's first attempts run at ~78 img/s (5.7-6.4 min/epoch), so
  it is now off by default (`--channels-last 1` to turn it back on).
- Part 2: `analyze.py` (done), preprocess.json (done), ensembles and fellow-eye blending (done). Still open: a val-only
  re-scorer so exp2's best.pt can be paired with later runs, TTA. Noise check done (exp7b): seed-to-seed
  difference 0.0001 on one pair; working `--noise-floor 0.007`.

## Experiments

Compare runs by best validation AUC; test numbers are recorded, never used to choose.

| Run | Change from baseline | Best val AUC (epoch) | Test AUC / sens / spec | Notes |
|---|---|---|---|---|
| exp1 | baseline: 384 px, no enhance, 30 epochs, defaults | 0.669 (28) | 0.638 / 87.0% / 23.3% | Underfitting: train loss 1.113 -> 1.037 (no-skill level ~1.11), train ~= val loss, val AUC still rising when the cosine LR reached zero. |
| exp2 | `--enhance` preprocessing (else = exp1) | 0.844 (26) | 0.829 / 88.8% / 48.8% | Starts learning lesions at epoch 3; train loss 1.112 -> 0.768, train ~= val loss (no overfitting); val AUC plateaus ~0.84 from epoch 21 as the LR decays. +0.175 val AUC: enhance is now the baseline. |
| exp3 | revamp defaults: `--target ordinal`, bilinear rotation with background fill (laptop, RTX 5060) | 0.844 (30) | 0.827 / 88.0% / 48.5% | Learning starts at epoch 2-3; train loss 0.338 -> 0.242 (no-skill 0.3385). Train AUC = val AUC at the end (0.847 vs 0.844): no overfitting. Still creeping up at epoch 30 as the LR reaches zero. ~2.2 min/epoch. |
| exp4 | as exp3 but `--target binary` (paired with exp3: same split, same everything else) | 0.846 (30) | 0.824 / 87.3% / 50.6% | Same curve shape as exp3. vs exp3: +0.002 val AUC = a tie (noise ~0.02). vs exp2: +0.002 = a tie, so the rotation fix made no measurable difference either. |
| exp5 | as exp3 but 512 px (`processed_512_enhanced`, `--size 512`) | 0.852 (27) | 0.828 / 88.0% / 47.9% | +0.008 val AUC over exp3 (same recipe at 384 px): small, under the ~0.02 noise, so "maybe slightly better", not proven. Same shape: plateau from epoch ~21 (0.848-0.852), train AUC = val AUC (0.847 vs 0.851). Epochs barely slower than at 384 px (~2.35 vs ~2.2 min), so 512 px stays. |
| exp6 | as exp5 but `--epochs 60` | **0.872** (48) | 0.855 / 87.5% / 57.2% | +0.020 val AUC over exp5, the first change since enhance to beat the ~0.85 ceiling (test AUC and specificity moved the same way). Slow climb until ~epoch 30, then steady gains as the LR decays; plateau from ~47. Train AUC now ahead of val (0.893 vs 0.871): the model starts fitting training eyes better, but val loss keeps falling, so no harmful overfitting. New base recipe: 512 px, 60 epochs (~2.4 h). |
| exp7 | as exp6 but images from `preprocess.py --enhance --tight-mask` (`processed_512_enhanced_tight`) | **0.888** (56) | 0.883 / 90.5% / 60.0% | Paired vs exp6 (analyze.py, same 5,268 val eyes): +0.016, 95% range +0.007 to +0.026, better in 100% of resamples, so beyond validation-sampling luck; seed noise not yet measured. Learned faster (0.874 at epoch 30 vs ~0.850 for exp6); plateau from ~45; train AUC 0.904 vs val 0.888, val loss lowest at the end. Some epochs ~2.85 min instead of ~2.35 (laptop heat). New base recipe: 512 px, tight mask, 60 epochs. |
| exp7b | exp7 repeated with `--seed 43` (noise check: same recipe, same split, different random init/order) | 0.888 | (not recorded) | Paired vs exp7: -0.0001, 95% range -0.007 to +0.006: the recipe reproduces almost exactly. One seed pair is a single sample, so use a cautious working noise floor of ~0.005-0.01 rather than 0.0001. With it, exp7 vs exp6 (+0.016) passes both checks: the tight mask is a confirmed improvement. |
| exp8 | as exp7 but `--pool gem` | **0.896** (50) | 0.889 / 91.8% / 60.8% | Paired vs exp7: +0.008, 95% range +0.001 to +0.015, better in 99% of resamples; above the 0.007 noise floor, so it passes both checks, narrowly. Learned much faster early (0.737 at epoch 2 vs 0.685; grade means separated by epoch 4); same healthy ending (plateau from ~43, train AUC 0.912 vs val 0.895, val loss lowest at the end). Steady 2.3 min/epoch with fans at max. New base recipe: 512 px, tight mask, 60 epochs, GeM. |
| exp9 | as exp8 but `--depth 34` (ResNet-34, 21.3M weights) | 0.896 (50) | 0.888 / 89.9% / 63.8% | A tie with exp8 (-0.0002 val AUC), with the same train-val gap (0.911 vs 0.895) and ~60% slower epochs (~3.7 vs ~2.3 min). Model size is not the limit at this data size; ResNet-18 stays. Its val specificity at 90% sensitivity ran a little higher (0.645 vs 0.618 at epoch 50), but at the same AUC that is noise in the operating point, not a better model. |

Reading exp2-exp4: three different recipes all end at ~0.845 val AUC after 30 epochs, with train AUC = val AUC and
the curve still rising slowly when the cosine LR hits zero. The ordinal target did not help on real photos (the
synthetic advantage did not carry over); it stays the default because it is a tie and also gives a severity score
and a real P(referable). The limit is the setup itself (384 px, 30 epochs, ResNet-18), not the target.

Heat maps of exp4 (laptop): confident false alarms sat on bright, lesion-like spots in eyes labeled grade 0
(possibly other conditions, e.g. laser scars or drusen, or label noise, which caps the reachable AUC); confident
misses had almost no evidence (peak 0.03), and on 9464_right the model ignored visible dark blots and looked at the
black area outside the eye, next to the bright enhancement rim. Hence the `--tight-mask` and `--pool gem` switches.

Reading exp5: resolution did not break the ~0.85 ceiling either. In every run since exp2, train AUC = val AUC at
the end and the curve flattens as the LR decays, so the model is not fitting even the training images better: it is
under-trained or too small, not overfitting. exp6 (60 epochs) confirmed it: training length was the limit.
exp7 (tight mask) beat exp6 in a paired comparison, so the base recipe is now 512 px + `--tight-mask` + 60 epochs. exp7b (seed repeat) matched exp7 within 0.0001, so the old "~0.02 noise" rule of thumb was too cautious: with
paired comparisons, differences of ~0.01 can be real. Working rule: a change counts if `analyze.py` check 1 passes
and the gain exceeds `--noise-floor 0.007`. exp8 (GeM) passed narrowly and joins the recipe. Next, one change each
against exp8: ~~`--depth 34`~~ (exp9: tie, slower), lower `--dropout`/`--wd`, more diverse training data (APTOS/DDR in the train split only). (On the tiny synthetic set GeM had looked slower than avg pooling; on real photos it was faster and better.)
Gains so far: +0.020 (60 epochs), +0.016 (tight mask), +0.008 (GeM), 0.000 (ResNet-34): the recipe is close to its
limit. Freeze candidate: the exp8 recipe (512 px, tight mask, 60 epochs, GeM, ResNet-18, ordinal target).

## Working with me

I'm learning ML through this project. Explain what you change and why, keep code well commented,
and don't swap my hand-written components for library versions without asking.

## Roadmap

1. Stage 1: from-scratch CNN for referable DR (this).
2. Stage 2: self-supervised pretraining (masked autoencoder, written from scratch) on pooled unlabeled retinal images.
3. Stage 3: fine-tune that model on multiple diseases (fundus multi-disease, glaucoma, OCT). Long-term goal: a
   general scan, one output per condition (as many as labeled data allows), plus an image-quality / "not
   something I recognize" check, since today's model sees every photo only as a DR score. Next small step on the
   way: report the predicted DR grade (0-4) from the ordinal model's cutpoints, not only referable yes/no.
4. Stage 4: external validation (e.g. Messidor-2) and comparison with published benchmarks.
5. Stage 5 (idea): a public demo page where people upload a photo and get the scores plus heat maps. Clearly an
   educational demo, not a diagnosis; store nothing; refuse images that are not usable retina photos.
