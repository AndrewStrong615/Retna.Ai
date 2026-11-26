"""
Small, hand-written building blocks for setting a screener up at a new clinic without grades,
shared by site_study.py (the measurements, E1-E4 of PREREG.md) and screen.py (the tool).

  - blend_batchnorm: "blended AdaBN" (method M2; Schneider et al. 2020): instead of replacing
    the EyePACS BatchNorm statistics by the clinic's, mix them, trusting the clinic more the
    more photos it gave. Protects a small or one-sided set-up batch (e.g. 5% sick) from
    pulling the statistics too far.
  - platt_fit / em_prevalence: estimate the share of sick eyes in a batch of UNGRADED photos
    from their scores (method M5; Saerens et al. 2002): turn scores into probabilities with a
    1-D logistic fit on EyePACS validation, then repeatedly re-weight them for a new prior.
  - rate_cutoff: method M4, refer the share of photos a clinic expects from its stated
    disease rate.
  - unsure_band / unsure_flags: the "unsure, refer" output: a band around the cutoff, sized on
    EyePACS validation, plus disagreement between the 8 TTA views.
None of these reads a grade from the clinic.
"""

import numpy as np
import torch


# ---------------------------------------------------------------- M2: blended AdaBN

def blend_batchnorm(adapted, source, n_setup, prior_n):
    """
    Mix each BatchNorm layer's statistics: w * EyePACS + (1 - w) * clinic, with
    w = prior_n / (prior_n + n_setup). prior_n is "how many photos the EyePACS statistics are
    worth": with prior_n = 0 this is plain AdaBN, with a huge prior_n nothing changes.
    The variance is mixed the same way (Schneider et al. 2020 do the same).
    `adapted` must be a copy whose statistics were re-measured on the clinic's photos
    (external_test.adapt_batchnorm); it is changed in place and returned.
    """
    w = prior_n / (prior_n + n_setup)
    src = [m for m in source.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    dst = [m for m in adapted.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    with torch.no_grad():
        for s, d in zip(src, dst):
            d.running_mean.mul_(1 - w).add_(w * s.running_mean.to(d.running_mean.device))
            d.running_var.mul_(1 - w).add_(w * s.running_var.to(d.running_var.device))
    return adapted


# ---------------------------------------------------------------- M5: how many are sick?

def platt_fit(score, sick, iters=50):
    """
    P(sick | score) = sigmoid(a * score + b), fitted by Newton's method on EyePACS validation
    (a 2-number logistic regression, written out). Returns (a, b).
    """
    x = np.asarray(score, dtype=np.float64)
    y = np.asarray(sick, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y)
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def em_prevalence(score, platt, train_prior, iters=200, tol=1e-8):
    """
    Saerens et al. (2002): the calibrated probability p_i assumes the training share of sick
    eyes (train_prior). If the batch has share q instead, each eye's probability becomes
        p_i' = q p_i / pi0 / (q p_i / pi0 + (1 - q)(1 - p_i) / (1 - pi0)),
    and q itself must equal the mean of the p_i'. Start at pi0 and repeat until it settles.
    Returns the estimated share of sick photos in the batch.
    """
    a, b = platt
    p = 1 / (1 + np.exp(-(a * np.asarray(score, dtype=np.float64) + b)))
    q = train_prior
    for _ in range(iters):
        num = q * p / train_prior
        den = num + (1 - q) * (1 - p) / (1 - train_prior)
        q_new = float(np.mean(num / den))
        if abs(q_new - q) < tol:
            return q_new
        q = q_new
    return q


# ---------------------------------------------------------------- M4: refer the expected share

def rate_cutoff(setup_scores, stated_rate, ref_spec, target_sensitivity=0.90):
    """
    A clinic says "about stated_rate of our patients are sick". At the EyePACS operating point
    (target_sensitivity, ref_spec) it should then refer
        share = target_sensitivity * r + (1 - ref_spec) * (1 - r)
    of its photos. The cutoff is the score above which that share of the (ungraded) set-up
    photos lies.
    """
    r = float(np.clip(stated_rate, 0.0, 1.0))
    share = target_sensitivity * r + (1 - ref_spec) * (1 - r)
    return float(np.quantile(np.asarray(setup_scores, dtype=np.float64), 1 - share))


# ---------------------------------------------------------------- "unsure, refer"

def unsure_band(ref_scores, cut, share=0.10):
    """Half-width of the band around the cutoff that holds `share` of the reference
    (EyePACS validation) eyes: the closest `share` of eyes to the cutoff are "unsure"."""
    return float(np.quantile(np.abs(np.asarray(ref_scores, dtype=np.float64) - cut), share))


def unsure_flags(score, cut, band, view_scores=None):
    """
    Unsure = the score sits within `band` of the cutoff, OR (when the 8 TTA views are given,
    shape (n_views, n_eyes)) the views disagree on which side of the cutoff the eye falls.
    """
    flag = np.abs(np.asarray(score) - cut) < band
    if view_scores is not None:
        sides = np.asarray(view_scores) >= cut
        flag |= sides.any(axis=0) & ~sides.all(axis=0)
    return flag


def unsure_readouts(sick, score, cut, flag):
    """How the unsure flag behaves on graded photos (H6): error rates inside and outside the
    flagged group, and sensitivity / specificity when unsure counts as refer."""
    sick = np.asarray(sick, dtype=bool)
    referred = np.asarray(score) >= cut
    wrong = referred != sick
    inside, outside = flag, ~flag
    refer2 = referred | flag
    return {"unsure_share": float(flag.mean()),
            "error_in_unsure": float(wrong[inside].mean()) if inside.any() else float("nan"),
            "error_outside": float(wrong[outside].mean()) if outside.any() else float("nan"),
            "sens_unsure_refer": float(refer2[sick].mean()) if sick.any() else float("nan"),
            "spec_unsure_refer": float((~refer2[~sick]).mean()) if (~sick).any() else float("nan")}


def _checks():
    """CPU checks (run by selftest.py)."""
    rng = np.random.default_rng(0)
    out = {}
    # Platt + EM recover a known prior from scores drawn as the model would give them.
    s_h, s_s = rng.normal(-1.5, 1, 20000), rng.normal(2, 1, 5000)
    score = np.concatenate([s_h, s_s])
    sick = np.r_[np.zeros(len(s_h)), np.ones(len(s_s))]
    platt = platt_fit(score, sick)
    batch = np.concatenate([rng.normal(-1.5, 1, 1900), rng.normal(2, 1, 100)])     # 5% sick
    out["EM recovers a 5% disease rate (+-2 points) from ungraded scores"] = abs(em_prevalence(batch, platt, 0.2) - 0.05) < 0.02
    # rate_cutoff refers the expected share.
    cut = rate_cutoff(batch, 0.05, ref_spec=0.8)
    out["rate cutoff refers 0.9 r + 0.2 (1 - r) of the photos"] = abs((batch >= cut).mean() - (0.9 * .05 + .2 * .95)) < 0.002
    # unsure band holds the stated share of reference eyes.
    band = unsure_band(score, 0.5, 0.10)
    out["unsure band holds 10% of reference eyes"] = abs(unsure_flags(score, 0.5, band).mean() - 0.10) < 0.002
    views = np.stack([score + rng.normal(0, 0.3, len(score)) for _ in range(8)])
    out["view disagreement only adds unsure eyes"] = (unsure_flags(score, 0.5, band, views) >= unsure_flags(score, 0.5, band)).all()
    # blended BatchNorm: w = 0 -> clinic stats, huge prior -> EyePACS stats.
    a, b = torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)
    a.running_mean.fill_(1.0); b.running_mean.fill_(3.0)
    m_src, m_ad = torch.nn.Sequential(a), torch.nn.Sequential(b)
    blend_batchnorm(m_ad, m_src, n_setup=100, prior_n=100)
    out["blended BatchNorm with prior = N sits halfway"] = torch.allclose(b.running_mean, torch.full((3,), 2.0))
    return out


if __name__ == "__main__":
    for name, ok in _checks().items():
        print("PASS " if ok else "FAIL ", name)
