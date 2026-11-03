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
- Not yet run: the GPU path (mixed precision with GradScaler).
- Real training needs an NVIDIA GPU. If none is available locally, prepare the run for a Kaggle Notebook
  (inputs in `/kaggle/input`, outputs to `/kaggle/working`).

## Working with me

I'm learning ML through this project. Explain what you change and why, keep code well commented,
and don't swap my hand-written components for library versions without asking.

## Roadmap

1. Stage 1: from-scratch CNN for referable DR (this).
2. Stage 2: self-supervised pretraining (masked autoencoder, written from scratch) on pooled unlabeled retinal images.
3. Stage 3: fine-tune that model on multiple diseases (fundus multi-disease, glaucoma, OCT).
4. Stage 4: external validation (e.g. Messidor-2) and comparison with published benchmarks.
