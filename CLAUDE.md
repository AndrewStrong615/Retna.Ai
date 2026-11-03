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

## Status

- Tested: preprocessing on synthetic images; patient split (no overlap); metric functions.
- **Not yet run:** `model.py` and `train.py` were written in an environment without PyTorch.
  First tasks: install deps (`torch torchvision scikit-learn pandas pillow`), run `python model.py`,
  build a tiny synthetic dataset and smoke-test `train.py` on CPU, then fix anything that breaks.
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
