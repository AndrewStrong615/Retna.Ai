# Retina-from-scratch

A from-scratch deep learning project to detect eye disease from retinal images.
**Stage 1 (done, see "Stage 1 result"):** classify referable diabetic retinopathy (grade >= 2) from
color fundus photos in the Kaggle EyePACS dataset (`trainLabels.csv`: columns `image,level`).

## Hard rules

- **No outside pretrained weights, ever.** No `pretrained=True`, no timm/torchvision/Hugging Face weights,
  no RETFound, no ImageNet checkpoints. PyTorch building blocks (Conv2d, BatchNorm, etc.) are fine.
  The one exception (Stage 2 on): weights this project pretrained itself, starting from random initialization,
  on this project's own photos (e.g. `pretrain.py`). Self-supervised pretraining must never use the labeled
  validation or test patients, so the comparison with Stage 1 stays fair.
- **Split by patient, not by image.** Images are named `{patient}_{left|right}`.
  Both eyes of a patient must land in the same split (see `split_by_patient` in `data.py`).
- **The test set is touched once**, at the end of a run. Model selection and the
  decision threshold are chosen on validation data only.
- Never commit dataset images or checkpoints to git. Follow each dataset's license terms.

## Files

- `preprocess.py`: crop black border, pad to square, resize, optional local-average-color enhancement. Run once.
  `--tight-mask` (off by default) uses each photo's real eye outline instead of a fixed circle, trims
  `--mask-margin` (2% of the size) at the edge, and with `--enhance` blurs only inside the eye, so no bright rim
  or dark crescents (seen where heat maps drifted on missed eyes). `--src` takes several folders (e.g. DDR's train,
  valid, test) into one output folder and refuses duplicate names. Writes `preprocess.json` into the output folder;
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
  `--accum N` (gradient accumulation): each batch goes through the GPU in N pieces, same recipe with less GPU
  memory (BatchNorm statistics per piece); `--accum 1` (default) is bit-identical to the old loop (checked on CPU).
  `--extra-labels CSV --extra-images DIR`: more labeled photos added to the TRAINING split only (validation/test
  eyes unchanged, so runs stay paired); refuses any validation/test patient and a folder preprocessed differently
  (preprocess.json); mean/std and background stay measured on the original training photos; recorded in config.json
  (`extra`) and `extra_split.csv`. Without the flags, training is bit-identical to before (checked on CPU).
  Several extra datasets at once: `--extra-labels A.csv B.csv --extra-images dirA dirB` (same order); with one
  extra dataset the training table is identical to before (checked on CPU against the old code's run).
  `--source-mix A B ...` (off by default = plain shuffling, bit-identical to before: checked on CPU with an extra
  dataset): share of every batch from each training source (`--images` first, then each `--extra-labels` in order),
  via the hand-written `SourceMixSampler` in data.py: same epoch length (same steps and LR schedule), each batch
  within one photo of the shares, each source's photos all used before any repeats. Recorded as `source_mix` in
  config.json (shares, natural shares, how often each photo is seen per epoch). The "balanced batches" camera idea:
  EyePACS validation cannot judge it, so it needs a new untouched external set, like exp17. Note: the no-skill loss
  and the cutpoints' starting values still follow the natural grade mix.
  `--ema D` (e.g. 0.9998; 0 = off, the old code path): a running average of the weights (hand-written
  `WeightAverage`, BatchNorm statistics and the loss's cutpoints included) is what gets validated, saved as best.pt
  and tested; training itself is unchanged. The live weights are validated too (`val_auc_live` in history.json)
  and their best epoch is written to `<out>/live/val_predictions.csv`, so `analyze.py <out>/live <out>` compares
  averaged vs live within ONE run (no seed noise).
  `--aug strong` (exp17; default `standard` = the old augmentation, same random numbers): adds camera differences to
  the training photos only: blur (p 0.3), down-up resize to 50-100% (p 0.3), colour jitter 0.4/0.4/0.3/hue 0.02,
  gamma 0.7-1.4 (p 0.5), JPEG quality 30-90 (p 0.3). The random gamma, down-up and JPEG steps are hand-written in data.py.
  Checked: `--aug standard` gives byte-identical val/test predictions and train losses to the old code on CPU
  (`CUDA_VISIBLE_DEVICES=-1`, `--workers 0`, 3 synthetic epochs; on the GPU even the old code differs run to run).
  bench.py at 512 px, 8 helpers: loading 229 img/s standard vs 151 strong (GPU alone ~175-180), so strong makes
  loading the limit, close to exp16's real 147 img/s. If exp17 shows data waits, try `--workers 12` (16 threads).
- `compare_external.py A B`: paired comparison of two `external_test.py` folders scored on the SAME photos (patient
  bootstrap of the differences in AUC, sensitivity and specificity, each system at its own cutoffs). Self-check:
  recomputes each folder's own numbers with the same resamples and refuses if they differ from its
  external_results.json. Reads finished outputs only, so it adds no look at a set.
- `adapt_site.py`: AdaBN as a deployment step for a NEW clinic (the IDRiD check S15 vs S15a showed it works).
  `adapt --tta-runs <run>_tta... --images <ungraded calibration photos> --out runs\site_X`: each model's 20
  BatchNorm layers re-measure their mean/variance on the clinic's photos (no grades, no weight trained; refuses fewer
  than `--min-photos` 200 and a second calibration into the same folder); saves `model_<i>.pt` (best.pt layout +
  `adapted`), `calibration_photos.csv` and `site.json` with the score scales and BOTH cutoffs exactly as fixed on
  EyePACS validation (no both-eyes, as external_test.py without `--pairs`). `score --site runs\site_X --images
  <photos> --out file.csv`: 8-view TTA with the adapted copies, standardized like validation, referred at both
  cutoffs. Reports nothing about accuracy (no grades): measuring AdaBN stays `external_test.py --adapt-bn`'s job.
- `convert_ddr.py`: DDR's grading labels (`DR_grading/{train,valid,test}.txt`, lines `<file> <grade>`) -> one CSV
  `image,level` for `--extra-labels`; drops grade 5 (ungradable) and label lines without a photo; refuses duplicate
  names and grades outside 0-5. All of DDR is training data (our validation/test stay pure EyePACS).
- `pretrain.py` (Stage 2): SimMIM-style masked image modeling, written from scratch, for the SAME network as
  Stage 1: hide 60% of the image in 32x32 squares (learned mask color), the ResNet stem+stages encode, a 1x1 conv
  + PixelShuffle redraws the pixels, L1 loss on hidden pixels only. Leaves out every image of the labeled
  validation/test patients (`--exclude-labels`, same `--split-seed` as train.py). Saves `encoder.pt` and
  `last.pt` every epoch (`--resume`), reconstruction pictures every `--example-every` epochs. `train.py --init
  <encoder.pt>` starts disease training from it (same `--depth`/`--pool` required; recorded in config.json).
- `vit.py` (Stage 2b building blocks, not yet wired into pretrain.py/train.py, never trained on real photos): a
  Vision Transformer (patch 16, 2-D sine/cosine positions, pre-norm blocks with hand-written multi-head attention,
  mean-pooled one-score head; ViT-Small default: dim 384, depth 12, 6 heads, 21.6M weights) and an MAE around it
  (75% of patches hidden and dropped from the encoder, 4-block decoder with a learned mask token, per-patch
  normalized pixel MSE on hidden patches only). `python vit.py` prints shapes and runs CPU checks (also in selftest):
  patchify round trip, 75% masking, attention equal to PyTorch's reference, MAE loss falling on a few images.
- `tta.py`: test-time augmentation for a finished run: scores each eye in the 8 exact orientations of a square
  (flips, 90/180/270 turns) and averages. Writes a run-like folder (`<run>_tta`) that analyze.py reads unchanged.
  Validation first (new 90%-sensitivity cutoff chosen there); `--split test` only once, after deciding; refuses
  to redo either. Its self-check: view 1 reproduces the run's own val AUC. `--eval-size N` scores the photos
  resized to N px instead of the training size (same weights; default folder `<run>_tta_s<N>`); the size is stored
  in tta.json, and the test pass and `external_test.py` reuse it (folders without it used the training size).
- `grade_report.py <run or run_tta>`: the 0-4 grade read off an ordinal run's learned cutpoints (median of the
  predicted grade distribution: how many boundaries have P(grade > k) >= 0.5), against the true grade: confusion
  table, exact / within-one agreement, quadratic weighted kappa (hand-written, the 2015 Kaggle metric) with a patient
  bootstrap range. Nothing chosen (cutpoints come from training); validation by default. Not for ensembles/both-eyes
  (standardized scores). train.py now also writes `predicted_grade` into ordinal runs' predictions CSVs
  (`CumulativeLogitLoss.predicted_grade` / `grade_probs`); every other column is unchanged.
- `quality_check.py` (Stage 3 step: "is this a photo I know how to read?"): `fit --run R --images <its photos>`
  measures the training photos' 512 pooled features (mean + covariance shrunk 10% toward a sphere) and sets limits
  on VALIDATION pictures (grades never read; `--quantile` 1% beyond each limit), saved as `R\quality_ref.pt`.
  `check --run R --images <folder> --out q.csv` flags per photo: unusual features (Mahalanobis distance, Lee et al.
  2018), unusual eye area (share of non-background pixels) and low detail (mean distance from the background colour
  inside the eye). Synthetic check: noise, blank and blurred pictures 0% ok, another synthetic clinic 86% ok. Not yet
  run on real photos; first real check: DDR grade-5 (ungradable) vs gradable photos.
- `rescore.py --run R --labels L --images I`: validation-only re-scorer: rebuilds the run's validation eyes as
  train.py does (`--split-after-drop` for the pre-revamp order of exp1/exp2), scores them once (plain view) and writes
  `<run>_rescored/val_predictions.csv` + rescore.json, so a run without predictions files (exp2) pairs with later
  runs in analyze.py. Reads old checkpoints (no arch/target/val_auc). Self-check: reproduces the run's own best val
  AUC (best.pt or history.json). No test option, on purpose; refuses to rescore twice.
- `check_labels.py`: authenticity check for a labels file from elsewhere: a finished run scores a random sample
  of its photos (plain view) and reports the AUC against the file's grades and the mean score per grade. Genuine
  labels land near the run's own val AUC; shuffled or mismatched ones near 0.5. Refuses validation/test patients.
- `external_test.py` (Stage 4): scores an external dataset ONCE with a frozen system (its `<run>_tta` folders), every
  choice re-derived from validation exactly as analyze.py does (shared `validation_choices`/`apply_choices`). Reports
  AUC with patient-bootstrap ranges at two cutoffs fixed on validation: standard (90% of sick validation eyes) and
  cautious (>= 90% sensitivity in 95% of validation bootstrap resamples). Messidor-2 formats: adjudicated grades CSV
  (`image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable`; ungradable photos left out) and the
  pairing file (`left;right`, one row per patient: the patient comes from it, not from the file name). Refuses a
  second scoring into the same `--out`, and photos preprocessed differently from the runs' training photos.
  Grade formats are detected from the columns: Messidor-2, APTOS (`id_code,diagnosis`) or plain (`image,level`).
  `--pairs` is optional: without it (APTOS) each photo is its own patient and the both-eyes blend is OFF, with
  the cutoffs then chosen on unblended validation scores (the Messidor-2 path is byte-identical to before).
  `--adapt-bn` (AdaBN, off by default): the photos are split by patient into 2 halves (seed 0); each half is scored
  by a model copy whose 20 BatchNorm layers re-measured their mean/variance on the OTHER half's images (plain view,
  no grades, no weights trained). Recorded as `adapt_bn` in external_results.json.
- `selftest.py`: one command, ~3-4 min on CPU: metrics, split, ordinal loss, saving, and a 2-epoch end-to-end run
  for both targets. Run it after every code change, before pushing.
- `bench.py`: speed test that times GPU-only training steps, image loading only (4 and 8 helpers) and both
  together, to find what limits training speed on a machine (~3-4 min; stop training first). `--aug strong` times
  the exp17 augmentation.
- `heatmap.py`: Grad-CAM written from scratch: for a finished run, colors where the model found evidence of
  disease on validation eyes (missed, false alarms, caught, cleared; refuses test images). Writes
  `<run>/heatmaps/*.png`. Each map is scaled to its own peak; the label's `peak` compares strength across eyes.
- `eye3d.py` + `eye3d_viewer.html` (3D eye view; checked on synthetic eyes only, not yet run on a real model):
  for a finished run, writes ONE web page `<run>/eye3d/viewer.html` (no internet, no library: hand-written WebGL) in
  which each chosen validation eye (same choice rules as heatmap.py; refuses test images) is an eyeball you can turn:
  the photo on the inside back wall, evidence in shades of red, numbered findings, click a spot to probe it.
  Heat = heatmap.py's Grad-CAM (region, 32 px cells) x LayerCAM from stage 2 (detail, 8 px cells; `--fine-stage 0` =
  plain Grad-CAM). Vessel sketch and "looks like" (on a vessel-like line / darker / brighter spot) are classical
  hand-written filters, NOT trained models, and describe pixels, not diagnoses. The 3D is a PROJECTION: one photo has
  no depth; the page assumes a round eye and that the photo covers `--cap-deg` 36 around the back pole (an assumption,
  adjustable in the page). Real depth would need OCT. The page embeds the photos: never commit it. Also writes
  `findings.json` (the same numbers as a table).
- `oct3d.py` + `oct3d_viewer.html` (TRUE 3D from OCT; checked on a synthetic volume only, no real scan yet): reads
  an OCT volume (folder of B-scan pictures, `.npy` [scan, depth, width], or Heidelberg `.vol`: parser written from
  the published layout, not yet tried on a real file; `.e2e`/DICOM not read), finds two surfaces per B-scan (ILM and
  the top of the photoreceptor/RPE band: the two strongest dark-to-bright edges, each as the best left-to-right path
  by hand-written dynamic programming, after Chiu et al. 2010; nothing learned), thickness map in um, thickened
  areas as numbered findings (default: above median + max(2.5 robust SD, 50 um) of THIS scan; no normal-eye
  database), and writes `<out>/viewer.html`: the measured surface as a 3D shape, red where thick, a B-scan sliding
  through the volume, click to measure; plus `layers.npz`, `summary.json`. `--demo` = built-in synthetic volume (known
  answer: surfaces within ~3 um). Pixel size comes from the .vol, else from `--width-mm/--length-mm/--depth-mm`
  (marked "assumed"). A .vol's own device surfaces are compared with ours (free check). The disease model reads
  fundus photos only: model heat reaches this page only through a registered photo (`--photo/--heat`, see
  register.py); an OCT model trained from scratch does not exist yet. Planned real data: OLIVES (Prabhushankar et al. 2022, Zenodo 7105232: DR/DME eyes, OCT volumes + near-IR
  fundus; check its licence before use; kept in `retina-data\olives`, never in the repo). Never commit scans or pages.
- `register.py` (photo + OCT fusion; checked on a synthetic pair only): lines a fundus photo up with an OCT volume by
  the vessels: photo vessel map (darker than surroundings, green channel) vs the OCT's vessel SHADOWS (mean of the
  bright band below the lower surface, seen from above); search over size (0.6-1.6x of `--photo-width-mm` 13) and
  turn (+-16 degrees, then fine), every shift at once by hand-written FFT normalized cross-correlation. Similarity
  transform only (no mirror, no bending). Built-in control: the same search with the scan's map upside down and
  transposed gives the chance level; TRUSTED only if fit >= 1.4x control and >= 0.25. Synthetic pair: under 1 px off,
  a photo with another vessel tree rejected. `oct3d.py --photo P [--heat H]` then lays the photo and the model's heat
  onto the measured surface (page: "Surface shows" / "Red shows", alignment check picture, model hot spots A, B..
  with the retina thickness under each); refuses an untrusted match unless `--force-fusion`. The heat is computed on
  the flat photo (`eye3d.py` now writes `<run>/eye3d/heat/<name>.png`); photo and heat must be the same preprocessed
  frame. `oct3d.py --demo` shows the fused cartoon pair (its heat is DRAWN, not from a model).
  Page shapes (switch under Display), three: "Scan only" (default since the scan-only rebuild) = NOTHING drawn: the
  whole photo in its own pixel colours (no tint, no lighting unless "Light the relief" is ticked) on a curved shell of
  radius 11 mm (the retina inside a 24 mm eye: 24 mm is the diameter), the OCT square raised by the MEASURED thickness
  x the relief slider, the OCT's own B-scan pixels on the square's four cut edges (first/last B-scan and the first/last
  column of every B-scan), a grey rim for the photo's edge and a 1 mm grid. "Surface shows": true photo colours / OCT
  seen from above (grey) / OCT depth in grey (thick = bright). Still ASSUMED: the curve itself, where the square sits
  on it, and all height OUTSIDE the square (the photo has no depth: it lies at the scan's median thickness). "Guess
  relief from photo" (off by default, tagged "estimated") raises dark vessels (<= 0.05 mm) and lowers the bright disc
  (<= 0.25 mm) from photo BRIGHTNESS: a guess for looking, never a measurement; a real disc cup or foveal pit shows
  only where an OCT scan covers it. "Textbook eye" (kept as an option) = a textbook-style cutaway of a STANDARD 24 mm eye, all
  hand-written WebGL (no three.js): the top-front quarter is cut away (y > 0 and z > -1.5 mm; "Cut the eye open"), the
  cut faces show sclera (white), choroid (dark red), retina (peach); cornea, anterior chamber, iris with pupil, pleated
  ciliary body, zonular fibres, lens, four rectus muscle stubs, optic nerve with central artery and vein. ALL of that is
  a DRAWING (coats drawn thicker than life), the same for every scan. The whole registered photo and the OCT patch are
  curved onto the inner back wall: a point l mm from the patch centre sits at angle l/11 from the back pole (11 mm =
  radius of the retina's back surface; distances along the wall kept), the lower surface lies ON the wall and the inner
  surface rises by the MEASURED thickness x the relief slider (1x anatomical to 5x; buttons for both ends); the drawn
  retina around the patch gets this scan's median thickness x the same relief, so the patch sits flush; the scan's own
  tilt is dropped. Assumed: eye size, the patch centred on the back pole, the nerve's side (chosen in the page).
  Part names with leader lines (two columns, pushed apart so they never overlap; hidden when the wall is in the way;
  "Hide names"). Buttons: Full cutaway / Zoom to OCT patch / Front view / Section from above. "Flat block" = exactly as
  scanned. All of this runs in the page (WebGL), looked at in a browser on the demo; selftest checks only the page's
  data and switches. Real data caveat:
  OLIVES pairs OCT with NEAR-INFRARED fundus pictures (as recalled; verify on download), not the colour photos the
  model was trained on: fine for testing registration, out of domain for the model's heat.
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
  `pd.read_csv(path, converters={"score": float})` to get it back exactly. For `--target binary` with
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
- Part 2: `analyze.py` (done), preprocess.json (done), ensembles and fellow-eye blending (done). TTA (`tta.py`) done. Val-only re-scorer
  (`rescore.py`) done in code; not yet run on exp2. Noise check done (exp7b): seed-to-seed
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
| exp10 | as exp8 but `--init runs\pre1\encoder.pt` (Stage 2a: SimMIM-pretrained start, 50 epochs on ~78k unlabeled photos) | **0.905** (50) | 0.903 / 90.3% / 65.2% | Paired vs exp8: +0.0086, 95% range +0.002 to +0.016, better in 99% of resamples, above the 0.007 noise floor: passes both checks, so self-supervised pretraining is a confirmed improvement (spec at 90% sens 61.8% -> 66.0%). Clear head start: 0.765 at epoch 1, 0.783 at epoch 2 (exp8: 0.737), 0.890 at epoch 29-30; plateau from ~41 (0.902-0.905). Train-val gap wider than exp8 (train AUC 0.934 vs val 0.903, exp8 0.912 vs 0.895), but val loss lowest near the end (0.2076 at 50) and flat after, so no harmful overfitting. Grade means swing epoch to epoch early (score offset shifts; ranking stable). Same speed as exp8 (~2.25 min/epoch). |
| exp11 | as exp10 but 768 px (`processed_768_enhanced_tight`, `--size 768`, `--init runs\pre1\encoder.pt`, batch 32, no `--accum`) | 0.909 (50) | 0.907 / 91.6% / 62.5% | Paired vs exp10: +0.0044, 95% range -0.002 to +0.011, better in 92% of resamples: fails both checks, a TIE. Spec at 90% sens lower (62.9% vs 66.0%). 768 px is not worth ~2.9x the compute; 512 px stays. TTA on exp11: 0.9090 -> 0.9140 (+0.0049), a 4th model where TTA helps. Same curve shape as exp10 (0.774 at epoch 1, 0.896 at 30, plateau 0.906-0.909 from ~44; train AUC 0.936 vs val 0.907). ~2.9x slower, not 2.25x: 67 img/s, 6.5 min/epoch (~6.6 h), some epochs 45-57 img/s with data waits of 13-17% (laptop busy/hot). |
| exp12 | as exp10 but `--extra-labels testLabels15.csv --extra-images processed_512_enhanced_tight_unlabeled` (+53,576 labeled EyePACS 2015 test photos, training only: 78,166 training photos, 3.2x) | **0.924** (48) | 0.916 / 89.3% / 73.1% | Paired vs exp10: +0.0195, 95% range +0.011 to +0.028, better in 100% of resamples, above the 0.007 floor: passes both checks, the biggest single-run gain since 60 epochs. More labeled data is the new base recipe. TTA: 0.9242 -> 0.9293 (+0.0052, a 5th model where TTA helps). Spec at 90% sens 73.1% vs 66.0%. Faster and higher all the way: 0.826 at epoch 1 (exp10 0.765), 0.901 by epoch 16 (exp10 needed ~40), plateau 0.920-0.924 from ~42. Val loss lowest 0.1896 at 48 (exp10 0.2076). Train-val gap 0.949 vs 0.923 (exp10 0.934 vs 0.903): slightly smaller, despite 3x more weight updates. No-skill 0.3389 (extra photos' grade mix). ~6.9 min/epoch at 193 img/s (~7 h). |
| exp12b | exp12 repeated with `--seed 43` (seed check at the 78k-photo scale; also an ensemble partner) | 0.927 (56) | 0.921 / 90.0% / 74.6% | Paired vs exp12: +0.0031, 95% range -0.002 to +0.008, better in 88% of resamples: a TIE, as a seed repeat should be. Second noise measurement: 0.003 (exp7b: 0.0001); the 0.007 floor stays cautious enough. Same curve (0.911-0.912 at epoch 30 in both, plateau from ~40, train AUC 0.947 vs val 0.927). TTA 0.9273 -> 0.9327 (+0.0055, a 6th model). Epochs 2, 6, 8 took 27-33 min with 59-67% waiting for data (laptop busy with something else: disk/CPU contention), otherwise ~7.2 min. |
| exp13 | as exp12 but `--wd 0.1` (stronger weight decay; default 0.05) | 0.928 (57) | 0.919 / 89.3% / 75.6% | Paired vs exp12: +0.0042, 95% range -0.001 to +0.009, better in 95% of resamples: fails both checks, a TIE (and inside the seed spread: exp12/exp12b 0.924/0.927). It did what weight decay should: train-val gap 0.015 (train AUC 0.943 vs val 0.928) vs 0.026 for exp12, lowest val loss 0.1869 (exp12 0.1896), spec at 90% sens 76.7% vs 73.1%; still creeping up at the end (0.927-0.928 over the last 6 epochs). Not an improvement under the rules; a harmless default to keep for future runs, not a new system. ~7.4 min/epoch. |
| exp14 | as exp13 but + DDR as a 2nd extra dataset (`--extra-labels testLabels15.csv ddr_labels.csv`; 12,522 gradable DDR photos; ~90,700 training photos) | 0.930 (59) | 0.920 / 89.6% / 74.4% | Paired vs exp13: +0.0013, 95% range -0.003 to +0.006, better in 69% of resamples: a tie on EyePACS validation, as expected (it cannot show DDR's camera diversity). Pre-set rule: NOT WORSE (lower bound -0.003 > -0.007), so DDR JOINS the recipe. No-skill 0.3695 (DDR's grade mix has more disease), so % of no-skill and train AUC are not comparable with earlier runs. Val loss lowest 0.1879 at 59, curve normal (0.913 at 27, plateau ~0.925-0.930 from 45). ~8.4 min/epoch (~8.5 h). |
| exp15 | as exp14 but DDR balanced to EyePACS's referable share (`ddr_labels_balanced.csv`: all grade 0-1 DDR photos + a seed-0 sample of the referable ones, 19.7% referable; ~8,590 DDR photos, ~86,750 training photos) | 0.930 (55) | 0.918 / 89.6% / 76.4% | Paired vs exp14: +0.0003, 95% range -0.004 to +0.005, better in 58% of resamples: a tie; pre-set rule NOT WORSE (lower bound -0.004 > -0.007), so balanced DDR is acceptable. No-skill 0.3399 (exp14 0.3695, EyePACS 0.339): the disease mix now matches EyePACS, as intended. Dropping ~3,900 sick DDR photos cost nothing visible on EyePACS validation. Whether it shrinks the camera shift needs a NEW external set (not Messidor-2). Slower: 141 img/s, ~10.5 min/epoch (GPU-bound, data waits 1%: heat or power mode). |
| exp16 | as exp15 but `--ema 0.9998` (averaged weights validated, saved and tested; live weights validated alongside) | 0.929 (55) | 0.926 / 90.2% / 74.6% | Deciding readout, averaged vs live weights of the SAME run: +0.0003, 95% range -0.002 to +0.003, better in 62% of resamples: check 1 NOT passed, so EMA does NOT join the recipe. Sanity vs exp15: -0.0014, range -0.006 to +0.002: not worse (lower bound > -0.007), barely. The averaged curve was much smoother (mean epoch-to-epoch change 0.0015 vs 0.0046 live; 0.0002 vs 0.0015 over epochs 45-60) and far ahead EARLY (0.845 vs 0.821 at epoch 1, 0.904 vs 0.888 at 16, 0.914 vs 0.899 at 30), but the cosine LR reaching zero already steadies the live weights, so at the end they meet (live best 0.9283 at 52). Useful for short runs or picking an epoch; nothing at the end of a full 60-epoch schedule. ~10.1 min/epoch, 147 img/s. |
| exp17 | as exp15 but `--aug strong` (camera-difference augmentation bundle) | 0.924 (52) | 0.917 / 89.9% / 71.7% | Readout A vs exp15: -0.0062, 95% range -0.011 to -0.001, better in 0% of resamples. analyze.py says tie (0.0062 < noise 0.007), but the pre-registered rule needed the lower bound > -0.007: FAILS, so exp17 is NOT scored on IDRiD and `--aug strong` does not join. Train/val gap small (0.939 vs 0.9235 at epoch 60). ~11.2 min/epoch, ~131 img/s, ~30% waiting for data (8 workers). |

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
Free tricks on exp8 (analyze.py, validation, paired vs exp8 alone at 0.896 / spec 61.8%):
- `--ensemble` exp8 + exp9: 0.901 (+0.005, range +0.002 to +0.008), spec 64.8%.
- `--both-eyes`: 0.917 (+0.021, range +0.016 to +0.026), spec 70.6%; w = 0.45 chosen on a smooth, flat peak
  (0.40-0.50 all ~0.917), so the choice barely matters. The biggest single gain since 60 epochs.
- `--ensemble --both-eyes` (exp8 + exp9, w = 0.40): **0.920** (+0.024, range +0.019 to +0.030), spec 71.6%.
DECISION (made on validation, before touching test): the frozen Stage 1 system is the exp8 + exp9 ensemble with
both-eyes blending (w = 0.40, cutoff for 90% sensitivity, all chosen on validation). Its official test result
comes from one run of `python analyze.py runs\exp8 runs\exp9 --ensemble --both-eyes --final-test`.

## Stage 1 result (frozen)

System: ensemble of exp8 + exp9 (512 px, tight mask, 60 epochs, GeM, ordinal target; ResNet-18 and ResNet-34),
each eye's score blended with its fellow eye's (w = 0.40), cutoff for 90% sensitivity; all chosen on validation.
Test set, evaluated once (5,268 eyes, 95% ranges by patient bootstrap):

| | AUC | Sensitivity | Specificity |
|---|---|---|---|
| Validation (choices made here) | 0.920 | 90.0% | 71.6% |
| **Test (official, touched once)** | **0.915** [0.900-0.928] | **89.9%** [87.6-92.0%] | **71.7%** [69.8-73.6%] |

Validation to test dropped only 0.005 AUC and specificity held, so the choices generalize within EyePACS.
From exp1: test AUC 0.638 -> 0.915; healthy eyes cleared at ~90% sensitivity 23% -> 72%. This is the benchmark
Stage 2 must beat. Not yet known: performance on other hospitals' photos (Stage 4).

## Stage 2a decision (made on validation, before touching test)

Candidates fixed in advance (two only, to limit optimism from trying many combinations on validation), with the rule
"if B is within ~0.003 of A, take A: one model is simpler and twice as fast":
- A: exp10 + `--both-eyes`: val AUC **0.924**, spec 73.1% at 90% sens (w = 0.40, flat peak 0.35-0.50 at 0.923-0.924).
- B: exp10 + exp9 `--ensemble --both-eyes`: val AUC 0.926, spec 74.0% (w = 0.40). +0.0018 over A: under the margin.
DECISION: the Stage 2a system is **A**: exp10 alone (pretrained ResNet-18 + GeM), each eye's score blended with its
fellow eye's (w = 0.40), cutoff for 90% sensitivity, all chosen on validation. It beats the frozen Stage 1 system on
validation (0.924 vs 0.920, spec 73.1% vs 71.6%) with one model instead of two. Official test result: one run of
`python analyze.py runs\exp10 --both-eyes --final-test` (recorded below, not used to choose anything).

TTA on exp10 (`tta.py`, 8 orientations, validation only; plain view reproduced 0.9047 exactly): 0.9115, paired vs
exp10 +0.0069, 95% range +0.004 to +0.010, better in 100% of resamples (check 1 passed). Check 2 (training luck) does
not apply: TTA reuses the SAME weights, so no seed noise enters the difference. Repeated on two other models
(plain views again reproduced their runs exactly): exp8 0.8961 -> 0.9014 (+0.0053, spec 61.8% -> 65.9%), exp9
0.8959 -> 0.9033 (+0.0074, spec 67.1%). Three models, three gains of +0.005 to +0.007: TTA is a confirmed,
general improvement and joins the standard tricks (tta.py, then --both-eyes). With `--both-eyes`: **0.929** (w = 0.40), spec 75.1% (vs 0.924 / 73.1%).
Not test-evaluated: bundle with the 768 px run (exp11) into one next system and one test pass, rather than spending
a test evaluation on each small trick.

## Stage 2a v2 decision (made on validation, before touching test)

Candidates fixed in advance (after the TTA check, before exp11 finished), same "within ~0.003, take the simpler" rule:
- C1: exp10 + TTA + `--both-eyes`: val AUC **0.929**, spec 75.1% (w = 0.40).
- C2: exp11 (768 px) + TTA + `--both-eyes`, ONLY if exp11 first passed the paired check vs exp10. It did not (tie), so
  C2 is out; its TTA-only val AUC (0.914 vs exp10's 0.9115, +0.0025) would not have cleared the margin either.
DECISION: the Stage 2a v2 system is **C1**: exp10 (512 px, pretrained ResNet-18 + GeM), each eye scored in the 8
orientations and averaged (tta.py), then blended with the fellow eye (w = 0.40), cutoff for 90% sensitivity, all
chosen on validation. Official test result: one run of `python tta.py --run runs\exp10 --images
...\processed_512_enhanced_tight --split test`, then `python analyze.py runs\exp10_tta --both-eyes --final-test`. Recorded below.

## Stage 2a v2 result (frozen)

Test set, evaluated once (5,268 eyes, 95% ranges by patient bootstrap):

| System (test, each touched once) | AUC | Sensitivity | Specificity | Models x views |
|---|---|---|---|---|
| Stage 1: exp8 + exp9, both eyes | 0.915 [0.900-0.928] | 89.9% [87.6-92.0%] | 71.7% [69.8-73.6%] | 2 x 1 |
| Stage 2a: exp10, both eyes | 0.921 [0.907-0.934] | 89.9% [87.6-92.2%] | 71.9% [70.0-73.7%] | 1 x 1 |
| Stage 2a v2 validation (choices made here) | 0.929 | 90.0% | 75.1% | 1 x 8 |
| **Stage 2a v2: exp10 + TTA, both eyes** | **0.925** [0.912-0.937] | **90.3%** [88.1-92.5%] | **73.5%** [71.6-75.3%] | **1 x 8** |

Validation to test dropped 0.004 AUC, as in the earlier systems: the choices generalize within EyePACS. Versus
Stage 2a on test: +0.004 AUC and +1.6 points of specificity at the same ~90% sensitivity (about 16 more healthy eyes
cleared per 1,000). The test ranges overlap, so the proof that TTA helps is the validation evidence (4 models,
+0.005 to +0.007 each), not this single test comparison. Cost: 8 forward passes per photo (still one model, ~0.1 s
per eye on the laptop GPU). This is the benchmark for the next improvement (more labeled data).


## Stage 2a result (frozen)

Test set, evaluated once (5,268 eyes, 95% ranges by patient bootstrap):

| | AUC | Sensitivity | Specificity | Models |
|---|---|---|---|---|
| Stage 1 frozen (exp8 + exp9, both eyes), test | 0.915 [0.900-0.928] | 89.9% [87.6-92.0%] | 71.7% [69.8-73.6%] | 2 |
| Stage 2a validation (choices made here) | 0.924 | 90.0% | 73.1% | 1 |
| **Stage 2a test (official, touched once)** | **0.921** [0.907-0.934] | **89.9%** [87.6-92.2%] | **71.9%** [70.0-73.7%] | **1** |

Validation to test dropped only 0.003 AUC, so the choices generalize within EyePACS. Versus Stage 1 on test: +0.006
AUC, same sensitivity, specificity +0.2 points, with ONE model instead of two (half the compute per photo). The test
ranges overlap, so on test alone this is "at least as good, simpler", not a proven gain; the proven gain is the paired
validation check (exp10 vs exp8: +0.0086, both checks passed). This is the new benchmark for later stages.

## Next: more labeled data (exp12)

Evidence: size (exp9) and resolution (exp11) gave nothing, the train-val gap keeps growing (exp11: 0.936 vs 0.907),
so the limit is the number of labeled photos. Source: `testLabels15.csv`, the published grades of the 53,576
EyePACS 2015 test photos (from the Kaggle mirror c7934597/resized-2015-2019-diabetic-retinopathy-detection,
`labels/`; keep it in `retina-data\data`, never in the repo). Checked: 53,576 unique `N_left/N_right` names, both
eyes for each of 26,788 patients, 0 patients shared with trainLabels.csv (so none with our val/test), grade mix
73.8/7.0/14.7/2.3/2.3% vs train 73.5/7.0/15.1/2.5/2.0%. The same photos were used UNLABELED in pretraining (pre1).
Authenticity check DONE (`check_labels.py`, exp10, 5,000 random photos, all 53,576 have a photo): AUC 0.9014 vs
exp10's own val 0.9047; at its val cutoff sens 91.0% / spec 64.3% (val: 90.0 / 66.0); mean score by grade
-1.32 / -0.68 / +1.66 / +3.86 / +5.15 (val: -1.34 / -0.99 / +1.64 / +4.04 / +5.03): genuine, same grading.
(Also an independent check that exp10 generalizes: 0.901 on 5,000 eyes it never saw labeled, in line with its
test AUC 0.903.) Plan: (1) done; (2) exp12 = exp10 recipe +
`--extra-labels ...\testLabels15.csv --extra-images ...\processed_512_enhanced_tight_unlabeled` (~78k training
photos, ~3x longer epochs); (3) paired vs exp10; (4) TTA + both eyes vs the frozen 0.929 (val) / 0.925 (test).

## Next system decision: candidates (fixed after exp12 finished, BEFORE its TTA / both-eyes numbers)

Precondition: exp12 must pass the paired check vs exp10 (`analyze.py runs\exp10 runs\exp12 --noise-floor 0.007`);
otherwise C1 (the frozen Stage 2a v2 system) stays. Then, rule "within ~0.003, take the simpler":
- D1: exp12 + TTA + `--both-eyes` (one model, like C1). Must beat C1's validation 0.929 to replace it.
- D2: ensemble exp12_tta + exp10_tta + `--both-eyes` (two models). Taken only if >= 0.003 above D1.
The winner gets one test pass (`tta.py --split test` for exp12, then `analyze.py ... --final-test`).

Results (validation): exp12 passed the paired check (+0.0195). D1 = **0.944** (w = 0.40, flat peak 0.35-0.45 at
0.944), spec 83.3% (vs C1 0.929 / 75.1%). D2 = 0.942, spec 81.6%: BELOW D1, so averaging in the weaker exp10 only
dilutes exp12. Both-eyes added +0.015 on top of exp12 + TTA (vs +0.017 on exp10): still large, a little smaller.
DECISION: the next system ("Stage 2 v3") is **D1**: exp12 (pretrained ResNet-18 + GeM, 512 px, trained on
78k labeled photos) + 8-view TTA + fellow-eye blend (w = 0.40), cutoff for 90% sensitivity, all chosen on
validation. Official test: `python tta.py --run runs\exp12 --images ...\processed_512_enhanced_tight --split
test`, then `python analyze.py runs\exp12_tta --both-eyes --final-test` (once). Recorded below.

## Stage 2 v3 result (frozen)

Test set, evaluated once (5,268 eyes, 95% ranges by patient bootstrap):

| System (test, each touched once) | AUC | Sensitivity | Specificity | Training labels |
|---|---|---|---|---|
| Stage 1: exp8 + exp9, both eyes | 0.915 [0.900-0.928] | 89.9% [87.6-92.0%] | 71.7% [69.8-73.6%] | 25k |
| Stage 2a v2: exp10 + TTA, both eyes | 0.925 [0.912-0.937] | 90.3% [88.1-92.5%] | 73.5% [71.6-75.3%] | 25k |
| Stage 2 v3 validation (choices made here) | 0.944 | 90.0% | 83.3% | 78k |
| **Stage 2 v3: exp12 + TTA, both eyes** | **0.936** [0.923-0.947] | **88.0%** [85.5-90.5%] | **83.3%** [81.7-84.7%] | **78k** |

AUC (threshold-free): +0.011 over Stage 2a v2, the largest system-level gain since Stage 1. Validation to test dropped
0.008 AUC, more than the earlier 0.003-0.005 (expected somewhat: the bigger the validation gain, the more of it can be
selection luck). The operating point moved: the cutoff chosen on validation for 90% sensitivity caught 88.0% of sick
test eyes (range 85.5-90.5%, so 90% is inside it) and cleared 83.3% of healthy ones. Part of the specificity jump
(+9.8 points) is that trade-off, so judge the gain by AUC, not by specificity alone. Open question, to settle on
VALIDATION and external data (Stage 4), never by retuning on this test set: a cutoff chosen for exactly 90% on 5,268
validation eyes lands ~2-3 points either side on new eyes; a screening program that needs >= 90% would pick a more
cautious cutoff (e.g. one that keeps 90% at the low end of a validation bootstrap).

## Stage 2 v4 decision (made on validation, before touching test)

Rule fixed BEFORE exp12b's results: the ensemble E1 (exp12_tta + exp12b_tta, `--ensemble --both-eyes`) replaces D1 only
if it beats D1's validation 0.944 (0.9441) by >= 0.003, i.e. >= 0.9471. Result: E1 = **0.9473** (w = 0.40, flat
0.35-0.45 at 0.947), spec 84.5% (D1: 83.3%). It passes by 0.0002: a pass under the rule, but the thinnest possible
one, so expect the test result to be about the same as D1's, not clearly better. Two equal-quality seeds were the
case where ensembling should help (unlike D2, where the weaker exp10 diluted exp12).
DECISION: the Stage 2 v4 system is **E1**: exp12 + exp12b (same recipe, seeds 42/43), each eye scored in 8
orientations per model, the two models' scores standardized on validation and averaged, blended with the fellow eye
(w = 0.40), cutoff for 90% sensitivity, all chosen on validation. Cost: 2 models x 8 views per eye.
Official test: `python tta.py --run runs\exp12b --images ...\processed_512_enhanced_tight --split test`, then
`python analyze.py runs\exp12_tta runs\exp12b_tta --ensemble --both-eyes --final-test` (once; exp12_tta's test
scores already exist from the v3 test pass and are reused unchanged).
Test-set budget: this is the 5th system scored on the same 5,268 test eyes (each once). Every look, even at different
systems, makes the test set a little less "fresh"; after this, the next confirmation should come from external data
(Messidor-2, Stage 4), not from this test set. Recorded below.

## Stage 2 v4 result (frozen, current best)

Test set, evaluated once (5,268 eyes, 95% ranges by patient bootstrap):

| System (test, each touched once) | AUC | Sensitivity | Specificity | Models x views |
|---|---|---|---|---|
| Stage 1: exp8 + exp9, both eyes | 0.915 [0.900-0.928] | 89.9% [87.6-92.0%] | 71.7% [69.8-73.6%] | 2 x 1 |
| Stage 2 v3: exp12 + TTA, both eyes | 0.936 [0.923-0.947] | 88.0% [85.5-90.5%] | 83.3% [81.7-84.7%] | 1 x 8 |
| Stage 2 v4 validation (choices made here) | 0.947 | 90.0% | 84.5% | 2 x 8 |
| **Stage 2 v4: exp12 + exp12b + TTA, both eyes** | **0.941** [0.929-0.951] | **88.6%** [86.2-91.0%] | **84.4%** [82.9-85.7%] | **2 x 8** |

Validation to test dropped 0.006 AUC. Versus v3 on test: +0.005 AUC, +0.6 points of sensitivity, +1.1 of specificity,
for twice the compute; the ranges overlap heavily, so "about the same or slightly better", as expected from a
validation pass by 0.0002. From exp1 (test 0.638, 23% of healthy eyes cleared) to 0.941 and 84%, all from scratch.
Pattern to note: both v3 and v4 landed BELOW 90% sensitivity on test (88.0%, 88.6%) with cutoffs chosen for 90.0% on
validation (earlier systems: 89.9-90.3%). Two systems are not proof of a bias, but it makes the cautious-cutoff
question (see v3) the first thing to settle on validation before Stage 4. This test set has now been used by 5
systems: the next confirmation comes from external data (Messidor-2).

## Stage 4 plan (fixed BEFORE any Messidor-2 photo is scored)

Data: Messidor-2 (ADCIS; 874 patients, 1,748 photos: 1,058 .png + 690 .jpg, all in `messidor2\IMAGES`), with the
adjudicated grades of Krause et al. 2018 (`messidor_data.csv`, Kaggle google-brain) and the ADCIS pairing file
(`messidor-2.csv`). Kept in `%USERPROFILE%\retina-data\messidor2`, never in the repo; research/education use only,
no redistribution; cite Krause et al. and the Messidor program partners. TEST ONLY: never trained on, never used to
choose anything.
System: the frozen Stage 2 v4 system (exp12_tta + exp12b_tta, both eyes w = 0.40), unchanged.
Primary result: AUC for moderate-or-worse DR (adjudicated grade >= 2, the definition the model was trained on),
95% range by patient bootstrap. Operating points: the standard and the cautious cutoff, both fixed on validation.
Secondary: AUC for "grade >= 2 OR macular edema" (common in published Messidor-2 benchmarks; our model never saw a
DME label, so expect it lower); per-grade referral shares; per-patient numbers.
Steps: (1) `preprocess.py --src ...\messidor2\IMAGES --dst ...\messidor2\processed_512_enhanced_tight --size 512
--enhance --tight-mask`; (2) `external_test.py --tta-runs runs\exp12_tta runs\exp12b_tta --images ... --grades ...
--pairs ... --out runs\messidor2_v4`, once; (3) record the numbers here, whatever they are.
Expectation (stated in advance): some drop from EyePACS is normal for a new camera/population; published from-scratch
models often lose several points of AUC. A large drop would point to camera/preprocessing differences, and the fix
would be more diverse TRAINING data (e.g. DDR), never tuning on Messidor-2. Recorded below.

## Stage 4 result: Messidor-2, external, scored once (frozen Stage 2 v4 system)

`runs\messidor2_v4` (external_results.json, external_predictions.csv). 1,744 gradable photos of 874 patients (4
ungradable left out), 457 referable (adjudicated grade >= 2, 26%). Cutoffs fixed on EyePACS validation before scoring:
standard +0.0321, cautious -0.0391 (on validation: sens 91.8%, spec 81.0%). 95% ranges by patient bootstrap.

| Messidor-2 (external) | AUC | Sensitivity | Specificity | Share of photos referred |
|---|---|---|---|---|
| Standard cutoff | **0.951** [0.937-0.963] | 97.6% [95.8-98.9%] | 62.5% [58.9-66.1%] | 53.2% |
| Cautious cutoff | (same) | 98.2% [96.9-99.3%] | 57.0% [53.2-60.5%] | 57.5% |
| For comparison: EyePACS test (v4, standard) | 0.941 [0.929-0.951] | 88.6% | 84.4% | |

Secondary: AUC for "grade >= 2 OR macular edema" 0.952 (the model never saw a DME label, yet ranks those eyes as
well). Referred by true grade (standard): grade 0 31.9%, 1 58.5%, 2 96.8%, 3 100%, 4 100%. Per patient (either eye
flagged): sens 96.3%, spec 62.9%.
Reading: (1) RANKING transfers: AUC 0.951 is not a drop from EyePACS; part of it is cleaner labels (adjudicated by a
retina-specialist panel, vs one EyePACS grader), so the two AUCs are not directly comparable. (2) The CUTOFF does not
transfer: scores run higher on Messidor-2 photos (other camera, other population), so the EyePACS cutoff refers
over half the photos: sensitivity rises to ~98% while specificity falls from ~84% to ~62% (31.9% of grade-0 eyes
referred). Discrimination generalizes, calibration does not: a real deployment would set its cutoff on a separate
local calibration sample from the new site (never on the test photos), or reduce the shift with more diverse
training cameras (DDR, APTOS). The cautious cutoff was not needed here (sensitivity already ~98%).
Context (published, not directly comparable: much larger training sets and ImageNet-pretrained networks): Gulshan
et al. 2016 AUC 0.990; Abramoff et al. 2016 (IDx-DR) AUC 0.980, sens 96.8%, spec 87.0%; Krause et al. 2018 AUC 0.986.
This project: 0.951 from scratch, 78k labeled photos, one laptop GPU.
Messidor-2 is now USED: never rescore it to choose anything; a future system can be scored on it once more only as
an additional report (each new system scored once), and its numbers must never steer training choices.

## Next: DDR as extra training data (exp14), plan fixed BEFORE any DDR run

Why: on EyePACS alone the recipe is near its limit (size, 768 px, weight decay: ties), more labeled photos gave the
biggest gain (+0.0195), and Messidor-2 showed that the model's scores shift with a new camera. DDR (Li et al. 2019:
13,673 photos, 147 hospitals in 23 Chinese provinces, many cameras; grade 5 = ungradable, dropped) adds both more
labels and camera diversity. TRAINING only; kept in `%USERPROFILE%\retina-data\ddr`, labels CSV in
`retina-data\data`, never in the repo; follow its license.
Steps: (1) `convert_ddr.py` -> `ddr_labels.csv`; (2) `preprocess.py --src <train> <valid> <test> --dst
...\processed_512_enhanced_tight_ddr --size 512 --enhance --tight-mask`; (3) `check_labels.py --run runs\exp12` on
DDR: a sanity check that photos and grades line up (expected below EyePACS's ~0.92, since other graders and
cameras; clearly above 0.5 is enough); (4) exp14 = exp13 (`--wd 0.1`) + DDR as a second extra dataset, ONE change
vs exp13; (5) `analyze.py runs\exp13 runs\exp14 --noise-floor 0.007`.
Decision rule (fixed now): validation is pure EyePACS, which cannot show DDR's main expected benefit (other cameras).
So DDR joins the recipe if exp14 is NOT WORSE than exp13 on validation: the paired difference's 95% range must not go
below -0.007 (the noise floor). A clear gain (both checks passed) is a bonus. If it is worse, DDR is left out.
Messidor-2 is used and must not steer this choice; a future system with DDR may be scored on it once, as a report.
Results: `check_labels.py --run runs\exp12` on 5,000 of the 12,522 gradable DDR photos (exp12 never saw DDR): AUC
**0.951**, at exp12's EyePACS cutoff sens 93.5% / spec 79.0%, mean score by grade -1.97 / +0.13 / +3.13 / +6.31 /
+7.54 (sample grade mix 50.5 / 4.8 / 35.7 / 1.7 / 7.2%: much more disease than EyePACS, very few "mild"; DDR's mild
eyes score near 0, above EyePACS's mild -0.99, a grading-convention difference). So: genuine labels, AND an informal
second external check of a single EyePACS model on Chinese hospital photos: ranking 0.951 like Messidor-2, with a
smaller cutoff shift (spec 79% vs 62.5% on Messidor-2). DDR is training data from now on, never a test.
exp14 vs exp13: +0.0013 [-0.003, +0.006] -> not worse -> DDR joins the recipe (base recipe now: exp13 + DDR).

## Report plan: does DDR reduce the camera shift? (fixed BEFORE scoring; report only, decides nothing)

The DDR decision is already made (exp14 vs exp13: not worse, DDR joins). This only measures what DDR was meant to
help and EyePACS validation cannot see: behavior on another hospital's photos. Two new single-model systems, each
scored on Messidor-2 ONCE with `external_test.py`, everything else equal:
- S13 = exp13 (no DDR) + 8-view TTA + both eyes; S14 = exp14 (+ DDR) + 8-view TTA + both eyes.
- w and both cutoffs come from each run's own EyePACS validation, as always.
Readouts: (1) specificity at the standard cutoff (the shift: v4 had 62.5%; a smaller shift = higher specificity at
similar sensitivity); (2) share of photos referred; (3) AUC (ranking). Hypothesis: S14 shows a smaller shift than S13.
Messidor-2 is then used by 3 systems (v4, S13, S14), each once; none of these numbers may steer training choices.

Results (each scored once; TTA on validation: exp14 0.9297 -> 0.9320, +0.0024, the smallest TTA gain so far):

| Messidor-2, single model + TTA + both eyes | AUC | Sens (standard) | Spec (standard) | Referred | Grade 0 referred |
|---|---|---|---|---|---|
| S13 (exp13, no DDR) | 0.952 [0.938-0.963] | 98.0% | 59.0% | 56.0% | 35.7% |
| S14 (exp14, + DDR) | **0.964** [0.952-0.972] | 99.8% | **43.0%** | 68.2% | 51.1% |

(Cautious cutoff: S13 98.5% / 52.6%; S14 100.0% / 35.1%. "grade >= 2 or DME" AUC: 0.953 / 0.964.)
HYPOTHESIS REFUTED for the shift: with DDR the EyePACS cutoff refers MORE Messidor-2 photos (68% vs 56%; half of
the grade-0 eyes), so the shift got worse, not better. But RANKING improved (+0.012 AUC). Likely reason: DDR is ~45%
referable vs ~19% in EyePACS, and its photos come from other cameras, so the model can learn "non-EyePACS-looking
photo -> more likely sick" (a source/prevalence shortcut); Messidor-2 photos then score higher overall. DDR's own
grading also puts "mild" eyes higher (see the DDR check). Practical reading: on a new site the cutoff must be set
locally anyway (calibration does not transfer), and then ranking (AUC) is what matters, which DDR improved.
These numbers were generated on Messidor-2, so they may suggest hypotheses (e.g. balance DDR's grade mix / prevalence
to EyePACS's when sampling it) but cannot confirm them: testing such a fix needs ANOTHER untouched external set.

## exp15 plan: DDR balanced to EyePACS's disease rate (fixed BEFORE the run)

Hypothesis (generated on Messidor-2, so Messidor-2 cannot confirm it): DDR's high referable share (~45% vs EyePACS
~19.7%) plus its different cameras teaches a source shortcut that pushes all non-EyePACS photos up. Fix to test:
keep all gradable non-referable DDR photos (grades 0-1) and a random sample (seed 0) of the referable ones, so that
DDR's referable share equals EyePACS training's 19.7% (`ddr_labels_balanced.csv`, made by a one-line pandas command).
exp15 = exp14 with `ddr_labels_balanced.csv` instead of `ddr_labels.csv`: ONE change.
Readouts: (1) EyePACS validation, paired vs exp14 (`analyze.py runs\exp14 runs\exp15 --noise-floor 0.007`): must
be NOT WORSE (95% range lower bound > -0.007), it throws away ~3,900 sick DDR photos; (2) the real question (does the
shift shrink?) can only be answered on a NEW external set never used before (e.g. IDRiD or APTOS, chosen and frozen
before scoring S14 and S15 on it once each). Messidor-2 must not be used to judge exp15.

## APTOS 2019 external check of exp15 (plan fixed BEFORE any APTOS photo is downloaded or scored)

Data: APTOS 2019 Blindness Detection (Kaggle; Aravind Eye Hospital, India), its TRAIN set only: 3,662 photos with
DR grades 0-4 (`trainLabels19.csv`, columns id_code,diagnosis, from the labels mirror; the photos from the official
competition `train_images`, not resized copies). Known grade mix ~49/10/27/5/8% (~41% referable). TEST ONLY, never
trained on, never used to choose anything; kept in `%USERPROFILE%\retina-data\aptos`, never in the repo; follow the
competition's rules. No patient/eye information, so no both-eyes blend; each photo is its own patient.
Systems (each scored ONCE, single model + 8-view TTA, no both-eyes, cutoffs from EyePACS validation):
- S14 = exp14_tta (all of DDR); S15 = exp15_tta (DDR balanced to EyePACS's referable share).
Primary readout: specificity at the standard cutoff and share of photos referred (the shift), read together with
sensitivity. Prediction: S15 shows the smaller shift (higher specificity at similar sensitivity). If S15 is no better,
the source/prevalence explanation of the Messidor-2 shift is weakened. Secondary: AUC (ranking).
Steps: (1) check counts only (3,662 labels, all photos present; no scoring); (2) `preprocess.py --src
...\aptos\train_images --dst ...\aptos\processed_512_enhanced_tight --size 512 --enhance --tight-mask`; (3)
`external_test.py` once for exp14_tta (`--out runs\aptos_s14`) and once for exp15_tta (`--out runs\aptos_s15`);
(4) record the numbers here, whatever they are. Do not run check_labels.py on APTOS: that would be an extra look.

Results (each system scored ONCE; `runs\aptos_s14`, `runs\aptos_s15`). Deviation from the plan: the grades came from the
competition's own `train.csv` (same columns `id_code,diagnosis`, 3,662 rows), not the labels mirror's `trainLabels19.csv`.
Counts checked first: 3,662 labels, 3,662 unique ids, all photos present; preprocessed with 0 failures. 1,487 referable
(40.6%). exp15's TTA on validation (made for this check): 0.9299 -> 0.9316 (+0.0016). No both-eyes (no pairing).

| APTOS 2019 (external), single model + TTA | AUC | Sens (standard) | Spec (standard) | Referred | Grade 0 referred | Grade 1 referred |
|---|---|---|---|---|---|---|
| S14 (exp14, all of DDR) | 0.965 [0.960-0.970] | 100.0% | 65.2% [63.2-67.2%] | 61.3% | 21.6% | 99.5% |
| S15 (exp15, DDR balanced) | **0.973** [0.969-0.978] | 100.0% | **72.0%** [70.1-73.9%] | 57.2% | 13.4% | 99.2% |

(Cautious cutoff: S14 100.0% / 62.2%; S15 100.0% / 70.3%. Cutoffs from EyePACS validation: S14 standard -0.0918,
S15 -0.0744.)
PREDICTION CONFIRMED: S15 shows the smaller shift: +6.8 points of specificity at the same (100%) sensitivity, the 95%
ranges do not overlap, 4 points fewer photos referred, and grade-0 false alarms nearly halved (21.6% -> 13.4%). This
supports the source/prevalence-shortcut explanation of the Messidor-2 shift (generated there, confirmed here on an
untouched set). Ranking also improved (+0.008 AUC, secondary). The shift itself is NOT gone: the EyePACS cutoff still
catches 100% of sick APTOS photos, so scores still run high on new cameras.
Where the remaining false alarms come from: almost all APTOS "mild" (grade 1) photos are referred (99%) by BOTH
systems, ~60% of S15's false alarms. Grade 0 alone: specificity 86.6% (S15), close to EyePACS. Likely a grading-
convention difference (as seen in DDR's check, where mild eyes scored near 0, far above EyePACS's mild) plus the shift,
not something to tune away on APTOS.
Caveats: APTOS has no patient IDs, so both eyes of one person count as two "patients" (ranges a little too narrow), and
APTOS is known to contain near-duplicate photos. APTOS is now USED by 2 systems: never rescore it to choose anything.
It chose nothing here: the exp15 recipe was already the base of exp16 before APTOS was scored.

## Scoring-size check (plan fixed BEFORE any photo is scored at another size; validation only)

Why: training crops a random 85-100% of each photo's area and stretches it to 512 px (`RandomResizedCrop` in
data.py), so lesions look 0-8% bigger in training (~4% on average) than at scoring time, when the whole photo is
resized to 512. Scoring slightly larger (upsampling the 512 px files) may undo that mismatch with the SAME weights
(Touvron et al. 2019, FixRes). Like TTA, no training and no seed noise: only check 1 (paired range) applies.
Models: exp12, exp12b, exp14 (their 512 px `_tta` folders are the baselines, already scored). Sizes: 544 and 576
only (544 ~ the average training zoom, 576 brackets it from above). `tta.py --run runs\expN --images ... --eval-size S`
(validation only), then `analyze.py runs\expN_tta runs\expN_tta_sS`.
Rule: a size joins the standard tricks only if its paired gain over 512 has a 95% range above 0 for ALL three models;
if both sizes pass, take the larger average gain. Choosing between 2 sizes on validation adds a little optimism, so
expect a small part of the gain to shrink on new eyes. The pre-registered APTOS check stays at 512 px either way
(its plan was fixed before this), and no test set or external set is scored for this check.

Results (validation, 8-view TTA, paired vs the same model at 512 px; spec at the 90%-sensitivity cutoff):

| Model | 512 px | 544 px: B - A [95% range] | 576 px: B - A [95% range] | Plain view only, 512 -> 544 |
|---|---|---|---|---|
| exp12 | 0.9293 / 74.3% | +0.0004 [-0.002, +0.003], 77.8% | -0.0000 [-0.004, +0.003], 77.9% | +0.0017 |
| exp12b | 0.9327 / 76.6% | +0.0016 [-0.001, +0.004], 76.5% | +0.0012 [-0.003, +0.005], 78.5% | +0.0016 |
| exp14 | 0.9320 / 76.8% | +0.0026 [0.000, +0.005], 78.1% | +0.0009 [-0.002, +0.004], 78.4% | +0.0024 |

VERDICT under the rule: NEITHER size joins (544 passes check 1 on one model of three, 576 on none). The direction
is consistent but small: 544 beats 512 on all three models (mean +0.0015 with TTA, +0.0019 for the plain view),
and 576 is smaller again, so the mismatch is real but near 544 and worth only ~0.002 AUC. TTA already removes
part of it (TTA gains shrink at 544). Kept as an option (`--eval-size`), not a standard trick; it could be
bundled with another small change later, never chosen again on these validation eyes alone.

## exp16 plan: weight averaging (--ema), fixed BEFORE the run

exp16 = exp15 + `--ema 0.9998` (average over ~5,000 updates, ~2 epochs): ONE change. Why: in the last 15 epochs
of exp12/exp13/exp15 val AUC still jumps 0.003-0.005 epoch to epoch, so the best epoch is partly a lucky pick;
averaged weights usually land in a flatter, steadier spot. Cost: one extra validation pass per epoch (~+20 s).
Readouts: (1) DECIDES: averaged vs live weights of the SAME run (`analyze.py runs\exp16\live runs\exp16`): EMA
joins the recipe if the 95% range is above 0 (check 1; no seed noise, training was identical; both best epochs are
picked on validation, and the jumpier live curve gets MORE max-of-noise optimism, so this is conservative for EMA).
(2) Sanity: exp16 vs exp15 (`--noise-floor 0.007`) must be NOT WORSE. Also watch: the averaged curve should be
smoother than the live one in history.json (`val_auc` vs `val_auc_live`).
Results: (1) averaged 0.9286 (epoch 55) vs live 0.9283 (epoch 52): +0.0003 [-0.002, +0.003], 62% of resamples:
EMA does NOT join the recipe (the option stays in train.py, off by default). (2) exp16 vs exp15: -0.0014
[-0.006, +0.002]: not worse. Curve: averaged much smoother and far ahead early (epoch 16: 0.904 vs 0.888), equal
by the end, because the cosine schedule's last epochs already average out the noise. Base recipe stays exp15.
Test (recorded, not used): 0.926 / 90.2% / 74.6% (exp15: 0.918 / 89.6% / 76.4%).

## exp17 + IDRiD plan: camera-robust augmentation, fixed BEFORE any IDRiD photo is downloaded or scored

Why: every external set so far (Messidor-2, APTOS) ranks well (AUC 0.95-0.97) but scores run HIGH on new cameras,
so the EyePACS cutoff refers too many healthy eyes (APTOS S15: spec 72.0% at 100% sens). Balancing DDR (exp15)
shrank the shift; the next lever is teaching the model to ignore camera differences: sharpness, compression,
noise, residual colour and contrast. Messidor-2 and APTOS are used, so judging this needs a NEW untouched set.

External set: IDRiD (Porwal et al. 2018; Eye Clinic, Nanded, India; one camera, Kowa VX-10 alpha, 4288x2848 px),
"B. Disease Grading": 516 photos (413 train + 103 test folders; BOTH are used as external test, NEVER trained on),
DR grade 0-4 and DME risk 0-2 per photo. Licence: CC BY 4.0 (IEEE DataPort DOI 10.21227/H25W98; free IEEE
login, the user downloads it); cite Porwal et al. Kept in `%USERPROFILE%\retina-data\idrid`, never in the repo.
No patient or eye information: each photo is its own patient, no both-eyes blend (as APTOS).
Expected size (from the dataset paper, checked in step 1): ~323 referable (~63%), ~193 not referable, only ~25 grade 1.
So specificity rests on ~193 photos: a 95% range of about +-6 points for one system. Two systems compared
UNPAIRED would need ~13 points to separate; hence the paired comparison below (same photos, resampled together).

The change, exp17 = exp15 + `--aug strong` (ONE switch, a bundle; its parts are NOT tested one by one, so no
extra looks at validation or IDRiD). Built in `data.py`, hand-written where torchvision has no transform, applied
to the training photos only, after the existing geometric steps:
- ColorJitter brightness 0.4, contrast 0.4, saturation 0.3, hue 0.02 (now 0.2 / 0.2 / 0.1 / 0).
- Random gamma 0.7-1.4 (p 0.5): different camera exposure curves.
- Gaussian blur, sigma 0.1-1.5 (p 0.3): softer optics, slight defocus.
- Down-then-up resize to 50-100% of the size (p 0.3): lower-resolution cameras.
- JPEG re-compression, quality 30-90 (p 0.3): APTOS and Messidor-2 .jpg photos vs EyePACS's.
`--aug standard` (default) must stay bit-identical to today (checked on CPU, as for --ema/--accum); selftest.py
gets a 2-epoch `--aug strong` run. Speed: `bench.py` before the run (blur/JPEG cost CPU; data waits are ~1% now).
Caveat stated now: training photos are `--enhance`d (local average colour subtracted), which already removes much
of the colour/illumination difference, so the expected gain is from sharpness/compression/noise more than colour.

Steps (in order):
(1) Code: `--aug strong`, selftest, bit-identity check of `--aug standard`, bench. PR, merged before the run.
(2) `compare_external.py A B`: a small paired comparison of two `external_test.py` folders scored on the SAME photos
    (patient bootstrap of the differences in AUC, sensitivity and specificity at each system's own standard cutoff).
    Self-check before IDRiD: on `runs\aptos_s14` vs `runs\aptos_s15` it must reproduce each folder's own
    external_results.json numbers (APTOS is used: this re-reads finished outputs and chooses nothing).
(3) Train exp17 (~10-11 h). Readout A, EyePACS validation: `analyze.py runs\exp15 runs\exp17 --noise-floor 0.007`;
    must be NOT WORSE (lower bound > -0.007), else stop: exp17 is not scored on IDRiD and IDRiD stays untouched.
    Then `tta.py --run runs\exp17` (validation only).
(4) IDRiD, only after (1)-(3): download; count check only (516 labels, all photos present, grade mix; no scoring);
    labels to plain `image,level` with a one-line pandas command (both CSVs: "Image name", "Retinopathy grade"). If
    the train and test folders reuse names (e.g. IDRiD_001 in both; preprocess.py refuses duplicates), prefix the
    test photos and their labels with `test_`. Preprocess: `preprocess.py --src ...\idrid\train ...\idrid\test --dst
    ...\idrid\processed_512_enhanced_tight --size 512 --enhance --tight-mask`; look at ~5 preprocessed photos for mask
    problems (pictures only, no scores).
(5) Score ONCE each: `external_test.py --tta-runs runs\exp15_tta ... --out runs\idrid_s15` and the same for
    `runs\exp17_tta` -> `runs\idrid_s17`. Cutoffs from each run's EyePACS validation, as always.
(6) `compare_external.py runs\idrid_s15 runs\idrid_s17`; record everything here, whatever it is.

Decision rule (fixed now): `--aug strong` JOINS the recipe if (A) EyePACS validation is NOT WORSE and (B) on IDRiD the
paired specificity difference S17 - S15 at the standard cutoff has a 95% range above 0, with S17's sensitivity still
>= 90%. Specificity can be bought by losing sensitivity, so a gain with sensitivity below 90% does NOT count.
If (A) passes and (B) does not, strong aug is left out (no benefit shown, and it costs CPU).
Secondary (report only): AUC difference (ranking); share referred; grade-0 referred; spec on grade 0 only;
AUC for "grade >= 2 or DME risk >= 1"; the target named earlier (>= 90% sens AND >= 85% spec at the EyePACS cutoff).
Prediction stated now: S17 has the higher specificity at ~100% sensitivity; S15 lands near its APTOS numbers
(spec ~70%), since IDRiD is another Indian clinic with a single non-EyePACS camera.
Budget: IDRiD is then USED by exactly these 3 systems (S15, S17, S15a below); never rescore it to choose anything.
Balanced batches across datasets (the other camera idea) is a separate change for later and needs yet another
untouched set.

### Amendment (2026-11-16, BEFORE any IDRiD photo is downloaded): a third system, S15a = exp15 + AdaBN

Why: the shift may sit largely in the BatchNorm statistics. Every BatchNorm layer normalizes with a mean and variance
measured on EyePACS photos; a new camera feeds it differently bright/contrasty features, so everything downstream (and
the score) moves. AdaBN (Li et al. 2016, "Revisiting Batch Normalization for Practical Domain Adaptation"): re-measure
those statistics on UNLABELED photos from the new site. No weight is trained, no grade is used. If it works, a new
clinic would only need a few hundred photos (no grades) before going live, and the EyePACS cutoff would transfer.
How (`external_test.py --adapt-bn`, default off = the old code path): the photos are split by patient into 2 halves
(seed 0). For each half, a fresh copy of the model re-measures its BatchNorm statistics (plain view, cumulative
average, dropout off) on the OTHER half's images only, then scores this half (8-view TTA as usual). So no photo is
scored by a model whose statistics saw it. Cutoffs and the TTA scale stay exactly as fixed on EyePACS validation.
Caveat: pooled numbers mix scores from two adapted copies (fitted on different halves); with ~258 photos per half
the two copies should be close. Score: `external_test.py --tta-runs runs\exp15_tta --adapt-bn ... --out
runs\idrid_s15a`, once, after S15 and S17 (step 5).
Rule (fixed now): AdaBN is called a working deployment step if `compare_external.py runs\idrid_s15 runs\idrid_s15a`
shows a specificity gain at the standard cutoff with a 95% range above 0 AND S15a still catches >= 90% of sick photos.
It is NOT a training change, so EyePACS validation cannot judge it and the S17 rule above is unchanged.
Prediction stated now: S15a has the higher specificity at a lower but still >= 90% sensitivity (scores move down).
Not scored on IDRiD: S17 + AdaBN (a 4th look); if both work, their combination waits for another untouched set.


## exp17 + IDRiD results (2026-11-18; each system scored ONCE)

- Readout A (step 3): exp17 failed not-worse (see the exp17 row), so S17 was never scored; IDRiD is used by S15 and S15a only. Deviation, stated openly: step 3's text says "IDRiD stays untouched" if exp17 fails, and the AdaBN amendment never updated that line. S15 vs S15a was run anyway (user approved 2026-11-18) because its rule and prediction were fixed before any IDRiD photo was downloaded; S17 stayed out.
- IDRiD prep: "B. Disease Grading" zip; train 413 + test 103 photos, test names prefixed `test_`; labels -> `idrid\idrid_labels.csv`
  (image,level). Count check: 516 labels = 516 photos, grades 0:168 1:25 2:168 3:93 4:62, 323 referable. Preprocessed
  512 px enhance + tight mask, 0 failures; 5 photos looked at, masks fine. No pairing: one photo = one patient.
- S15 (`runs\idrid_s15`): AUC 0.952 [0.934, 0.966]; standard cutoff sens 100.0%, spec 39.9% [32.7, 46.7], refers 85.1%.
  The worst camera shift seen so far (APTOS spec 72%): ranking holds, the EyePACS cutoff does not.
- S15a = + AdaBN (`runs\idrid_s15a`): AUC 0.968 [0.955, 0.978]; standard cutoff sens 96.3% [94.3, 98.2], spec 75.1% [68.7, 81.1],
  refers 69.6%. Grade 0 referred 56.5% -> 20.8%; grade 2 100% -> 93.5%.
- `compare_external.py runs\idrid_s15 runs\idrid_s15a`: AUC +0.0161 [0.006, 0.028]; standard sens -3.7 points [-5.7, -1.8],
  spec +35.2 points [+28.6, +42.0]. Rule: spec gain range above 0 YES, sens >= 90% YES => AdaBN IS a working deployment step.
  The stated prediction (higher spec at lower but >= 90% sens) came true.
- Meaning: most of the camera shift sits in the BatchNorm statistics. A new clinic needs a few hundred UNGRADED photos
  before going live. IDRiD is now used up for choosing anything.

## Working with me

I'm learning ML through this project. Explain what you change and why, keep code well commented,
and don't swap my hand-written components for library versions without asking.

## Roadmap

1. Stage 1: from-scratch CNN for referable DR. Done: test AUC 0.915, sens 89.9%, spec 71.7% (see above).
2. Stage 2: self-supervised pretraining (masked autoencoder, written from scratch) on pooled unlabeled retinal images.
   Stage 2a (chosen): SimMIM-style masked image modeling with the Stage 1 ResNet-18 + GeM (`pretrain.py`), so the
   only difference from exp8 is the starting weights. Pool: the unused EyePACS `test` folder (~53k) plus the
   training split's images (labels unused); validation/test patients excluded. Plan: preprocess `data\test` with
   `--enhance --tight-mask` at 512 px, pretrain ~50 epochs (~6-7 min/epoch, ~6 h), fine-tune with the exp8 recipe
   and `--init`, compare paired with exp8 (`analyze.py --noise-floor 0.007`), then with the tricks vs the frozen
   0.915. pre1 done (laptop, GPU): 50 epochs, 8.0 min/epoch (162 img/s, ~6.7 h); masked-pixel L1 loss
   0.495 -> 0.401 (epoch 2) -> 0.356 (5) -> 0.339 (10) -> 0.320 (30) -> 0.309 (50), still easing down at the end
   (no collapse, no blow-up). Watch out: if `train.py`/`pretrain.py` print `Device: cpu`, the NVIDIA driver is not
   visible (check `nvidia-smi`, power mode, restart); stop at once. Fine-tuned as exp10 (see the table): confirmed better than exp8. With both eyes: val 0.924, TEST 0.921 / 89.9% / 71.9% with one model (see "Stage 2a result"); with TTA too
   (Stage 2a v2): TEST 0.925 / 90.3% / 73.5%. With 3x the labeled photos (exp12, Stage 2 v3): TEST 0.936 / 88.0% /
   83.3%. Two seeds of that recipe (Stage 2 v4): TEST 0.941 / 88.6% / 84.4%, the current best.
   External (Messidor-2, Stage 4): AUC 0.951; at the EyePACS cutoff sens 97.6%, spec 62.5% (see "Stage 4 result").
   Stage 2b (later): a Vision Transformer + MAE, written from scratch, once more unlabeled photos exist.
   Building blocks done in code (`vit.py`, CPU checks only); next: an MAE mode in pretrain.py and a ViT option in
   train.py, then a GPU run.
3. Stage 3: fine-tune that model on multiple diseases (fundus multi-disease, glaucoma, OCT). Long-term goal: a
   general scan, one output per condition (as many as labeled data allows), plus an image-quality / "not
   something I recognize" check, since today's model sees every photo only as a DR score. Predicted DR grade
   (0-4) from the ordinal cutpoints: code done (`grade_report.py`), not yet run on a real model.
4. Stage 4: external validation (e.g. Messidor-2) and comparison with published benchmarks.
5. Stage 5 (idea): a public demo page where people upload a photo and get the scores plus heat maps. Clearly an
   educational demo, not a diagnosis; store nothing; refuse images that are not usable retina photos.
