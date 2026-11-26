# Pre-registration: label-free clinic set-up at real disease rates

Written 2026-11-26, committed BEFORE any real photo is scored by `site_sim.py`; the git date of this file is
the registration time. A rule written here is not changed after its experiment starts. If something must
change, the change is added below as a dated amendment with its reason, and the result is reported both ways
where possible.

Full plan (phases, datasets, paper): Claude Doc "Plan: the self-calibrating DR screener".

## Question

On IDRiD, AdaBN (BatchNorm statistics re-measured on the clinic's ungraded photos) raised specificity at the
EyePACS cutoff from 39.9% to 75.1%, at 96.3% sensitivity. But 63% of IDRiD's photos are sick; screening clinics
see 5-20%. Does AdaBN still put a clinic on the right operating point at real disease rates, or does part of
its effect come from the disease rate of the set-up photos rather than from the camera?

## Standing choices (apply to every experiment)

1. **Knobs** (any setting a method has: blending weights, an unsure band, quality limits) are set only on EyePACS
   validation and on "fake cameras" (validation photos passed through a fixed camera change). No real external
   set, fresh or used, ever sets a knob. (Plan question 2; andrew gave no answer, so this recommended default
   applies.)
2. **Used external sets** (APTOS, Messidor-2) may be scored again for the disease-rate grid (E2) as REPORT ONLY,
   with every method and knob frozen first; they choose nothing. IDRiD is not rescored. (Plan question 3; no
   answer, recommended default.)
3. No science fair (andrew, 2026-11-26), so no fair forms gate the work.
4. The EyePACS test set is not touched by any experiment here.

## E0: disease rate alone, no camera change (EyePACS validation only)

Script: `site_sim.py`. Model: **exp15** (`runs/exp15/best.pt`), the base recipe that was scored on IDRiD.

Design, fixed now:
- Validation patients (5,268 eyes) split into two halves with `external_test.two_halves` (seed 0), both eyes
  of a patient together. Half 0 = set-up pool; half 1 = scored clinic (all its sick eyes + a fixed random 1,000
  of its healthy eyes, seed 0).
- Pretend clinic: a set-up batch of **N = 200** pool eyes with sick share **p = 5, 10, 20, 40, 60%**
  (sick = grade >= 2), drawn without replacement; **20 draws** per p. Grades only build the batch.
- **Plain view** (no TTA) on CPU, fp32. The cutoff is the plain-view 90%-sensitivity cutoff, recomputed on all
  5,268 validation eyes from this CPU's scores. Self-check before anything else: validation AUC within 0.002 of
  the run's `val_predictions.csv` and score correlation >= 0.999, else the run stops.
- Methods: M0 nothing; **M1 AdaBN** (all BatchNorm layers, cumulative average, as `external_test.adapt_batchnorm`);
  M3 score reset (scores shifted and scaled so the set-up batch's mean and spread match all validation eyes').
- Fake camera, fixed now: gamma 1.3, then Gaussian blur sigma 1.0, then JPEG quality 40, applied to set-up and
  scored photos; p = 20%, 20 draws.
- Readouts per draw: sensitivity, specificity, AUC, share referred on the scored clinic at the fixed cutoff;
  shifts are measured against M0 on the same scored clinic.

Deviations from the plan doc, decided now for cost on a 4-core CPU (about 20 photos/s): 20 draws instead of 50,
N = 200 only (the N curve is E3), plain view instead of 8-view TTA, a 1,000-eye healthy sample in the scored
clinic. M3 is added because it costs nothing extra.

Hypotheses and rules (computed by `site_sim.verdicts`, exactly as below):
- **H0a**: at p = 20% (the training mix), AdaBN changes the median sensitivity and the median specificity by less
  than 2 points each (|median shift| < 0.02).
- **H0b**: as p rises, AdaBN's sensitivity falls and specificity rises: Spearman rho(p, shift) < 0 for
  sensitivity and > 0 for specificity over all 100 draws, AND the median sensitivity shift at 60% is at least
  2 points below that at 5%, AND the median specificity shift at 60% at least 2 points above that at 5%.
- **H0c**: with the fake camera, AdaBN at least halves the distance from the clean operating point
  (|sens shift| + |spec shift|, median over draws, vs M0 on fake photos). Not testable if the fake camera moves
  M0 by less than 5 points.
- **Stop rule**: if AdaBN's median sensitivity at any p <= 20% is below 85%, plain AdaBN is called UNSAFE for
  screening, and blended AdaBN (M2) and AdaBN + disease-rate estimate (M5) become the main methods of E1-E4.
- Secondary, no verdict: M3's numbers; spread between draws (5th-95th percentile); share of draws with
  sensitivity >= 90% and >= 85%.

What a result means: H0b passing says part of IDRiD's gain (and its 3.7-point sensitivity loss) can come from
its 63% sick share alone, so AdaBN must be judged at realistic rates (E2). H0b failing says the disease rate of
the set-up photos does not matter at N = 200, which makes AdaBN simpler to recommend.

## E1-E4, S1-S7

Registered in full in the plan doc before their data is downloaded; copied here, with any E0-driven change marked
as an amendment, before each one starts.

## Amendments

**1 (2026-11-26, before any real photo was scored).** The laptop, the only place exp15's weights exist, is out
of reach. If E0 is run on Kaggle (`kaggle_e0.ipynb`) instead, the model is **kexp8**: a fresh run of the exp8
recipe (512 px, enhance + tight mask, 60 epochs, GeM, ResNet-18, ordinal target, random start, no pretraining
and no extra labels; laptop exp8 reached val AUC 0.896), trained on Kaggle's EyePACS copy with the same split.
Everything else (design, hypotheses, rules, fake camera) is unchanged. It runs on the GPU in fp32 instead of the
CPU, so the self-check compares with kexp8's own val_predictions.csv. Consequence for the reading: kexp8 is not
the model scored on IDRiD, so E0 then speaks about AdaBN on this project's recipe, not about S15 itself; if the
laptop comes back, E0 on exp15 can be run as registered and both are reported.
