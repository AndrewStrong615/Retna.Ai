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
- `train.py`: training loop (AdamW, warmup + cosine LR, mixed precision, pos_weight for class imbalance),
  validation AUC and specificity at a target sensitivity, best-checkpoint saving, final test evaluation.
- `make_synthetic_data.py`: tiny fake EyePACS-style dataset (cartoon eyes, obvious lesions) for CPU smoke tests.
  Its docstring has the smoke-test commands.
- `requirements.txt`: dependencies, with the versions the smoke test passed on.
- `kaggle_train.ipynb`: the GPU run on Kaggle: GPU check, clone this repo, find the data, preprocess,
  2-epoch sanity check, full run, results.

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

## Experiments

Compare runs by best validation AUC; test numbers are recorded, never used to choose.

| Run | Change from baseline | Best val AUC (epoch) | Test AUC / sens / spec | Notes |
|---|---|---|---|---|
| exp1 | baseline: 384 px, no enhance, 30 epochs, defaults | 0.669 (28) | 0.638 / 87.0% / 23.3% | Underfitting: train loss 1.113 -> 1.037 (no-skill level ~1.11), train ~= val loss, val AUC still rising when the cosine LR reached zero. |
| exp2 | `--enhance` preprocessing (else = exp1) | 0.844 (26) | 0.829 / 88.8% / 48.8% | Starts learning lesions at epoch 3; train loss 1.112 -> 0.768, train ~= val loss (no overfitting); val AUC plateaus ~0.84 from epoch 21 as the LR decays. +0.175 val AUC: enhance is now the baseline. |

## Working with me

I'm learning ML through this project. Explain what you change and why, keep code well commented,
and don't swap my hand-written components for library versions without asking.

## Roadmap

1. Stage 1: from-scratch CNN for referable DR (this).
2. Stage 2: self-supervised pretraining (masked autoencoder, written from scratch) on pooled unlabeled retinal images.
3. Stage 3: fine-tune that model on multiple diseases (fundus multi-disease, glaucoma, OCT).
4. Stage 4: external validation (e.g. Messidor-2) and comparison with published benchmarks.
