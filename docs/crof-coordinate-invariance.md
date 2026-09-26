# Is ROF's between-run evidence coordinate-dependent? (September 2026)

ROF is a ratio of Euclidean norms in the model's latent coordinates. The observable and
unobservable directions move with any invertible change of coordinates s' = Ts, but the
fraction does not: with C = [1, 0], R = [1, 1] and T = diag(a, 1), ROF goes from 1/2 to
1/(1+a²). Expectations for the test below were written down before any number was
computed. The theory note (`docs/fisher-support-note.tex`, Remark "Coordinates") now states
the dependence.

## Measure

Per checkpoint, the state covariance Σ is estimated from 1024 val-set states, collected the
way the Jacobian states are (posterior warmup, [h, z] at t = 5). The same hard ROF construction
(1e-3 threshold) is then applied to O_H Σ^(1/2) and Σ^(1/2) Rᵀ.

- wfull (symmetric square root of Σ): unchanged under any invertible linear change of
  coordinates. It measures the share of the linearised reward's variance, over how the state
  actually varies, that lies in observable directions.
- wdiag (diag of per-coordinate standard deviations): unchanged under per-coordinate rescaling
  only. Secondary.

Code: `eval_metrics.py --whiten`. Every other field is unchanged, and the sealed jac_rof and
jac_rof_bad reproduce the stored sweeps (19 of 7200 values differ, each by at most 0.0002).
Files: `results/metrics_w_<run>_seed{12345,1,2}.txt`. Scorer: `results/score_whiten.py`.

## Levels (seed-averaged, epochs ≥ 50)

| run | bad ROF | bad wfull | bad wdiag | good ROF | good wfull | good wdiag |
|---|---|---|---|---|---|---|
| sp909 | 0.279 | 0.918 | 0.471 | 0.228 | 0.873 | 0.540 |
| sp1010 | 0.237 | 0.897 | 0.496 | 0.214 | 0.861 | 0.583 |
| f25_909 | 0.304 | 0.898 | 0.479 | 0.260 | 0.886 | 0.502 |
| f25_1010 | 0.291 | 0.899 | 0.468 | 0.266 | 0.886 | 0.516 |
| f75_909 | 0.292 | 0.906 | 0.469 | 0.259 | 0.866 | 0.538 |
| f75_1010 | 0.263 | 0.901 | 0.467 | 0.241 | 0.866 | 0.539 |
| scr_909 | 0.340 | 0.904 | 0.453 | 0.289 | 0.896 | 0.467 |
| scr_1010 | 0.308 | 0.889 | 0.443 | 0.285 | 0.884 | 0.475 |
| h128_909 | 0.367 | 0.913 | 0.459 | 0.321 | 0.872 | 0.504 |
| h128_1010 | 0.347 | 0.915 | 0.526 | 0.301 | 0.872 | 0.593 |
| z8_909 | 0.259 | 0.916 | 0.526 | 0.236 | 0.887 | 0.604 |
| z8_1010 | 0.260 | 0.933 | 0.469 | 0.225 | 0.890 | 0.554 |
| v7 | 0.254 | 0.895 | 0.514 | 0.208 | 0.853 | 0.586 |
| zonly3080 | 0.217 | 0.882 | 0.572 | 0.231 | 0.883 | 0.581 |

## Results

| question | ROF | wfull | wdiag |
|---|---|---|---|
| rank correlation with ROF, bad-pool level, 12 runs | 1 | -0.09 | -0.43 |
| h128 offset from the sp pair (bad pool) | +0.098 | +0.007 | +0.009 |
| as a fraction of the 8 screening runs' spread | 0.95 | 0.24 | 0.16 |
| z8 offset (bad pool) | +0.001 | +0.017 | +0.014 |
| z-only 3080 vs V7, good pool, epochs ≥ 400 | 0.285 vs 0.196 | 0.881 vs 0.836 | 0.621 vs 0.582 |

Screening runs by bad-pool level: ROF scr_909, scr_1010, f25_909, f75_909, f25_1010, sp909, f75_1010, sp1010; wfull sp909, f75_909, scr_909, f75_1010, f25_1010, f25_909, sp1010, scr_1010.

Within runs, the rank correlation across checkpoints (epochs ≥ 50) between the ROF and wfull
good-pool series has median +0.53 over the twelve runs, with a range of -0.48 to
+0.83.

The whitened levels still separate runs. Their between-run spread is 6 to 9 times the
standard error of a three-seed mean (ROF: over 80). The runs are separated in a different
order, so this is not whitening flattening everything into noise.

## Reading

- The data-family ordering of bad-pool levels, and the h128 offset, do not survive a
  change of coordinates. They describe how each model scales its latent coordinates and
  should not be offered as properties of the modelled process. The screening reading and the
  per-architecture normalisation argument rest on them.
- The z-only intervention still raises the fraction in both invariant variants, with the gap
  about halved.
- Within-run trends partly survive, but not reliably run by run.
- Weighted by how the state varies, 85–93% of reward sensitivity lies in observable
  directions in every run. Native levels of 0.2–0.37 should not be read as "most of the
  reward is unobservable".


## Addendum: the Reacher–LunarLander contrast

The draft's domain contrast is the within-run Spearman correlation between pool-averaged ROF
(0.5·good + 0.5·bad) and MA-7 MPC return, over the same eight training seeds on both tasks.
The same statistic is computed here with the whitened measure. Expectations were written
down before the sweeps ran: the contrast survives if Reacher stays positive in all eight
runs and LunarLander still spans zero.

| seed | Reacher ROF | Reacher wfull | Reacher wdiag | LunarLander ROF | LunarLander wfull | LunarLander wdiag |
|---|---|---|---|---|---|---|
| 101 | +0.60 | -0.56 | +0.40 | +0.32 | +0.10 | +0.27 |
| 202 | +0.49 | -0.62 | +0.18 | -0.55 | +0.48 | -0.04 |
| 303 | +0.87 | -0.73 | +0.75 | -0.31 | +0.77 | +0.20 |
| 404 | +0.60 | -0.58 | +0.14 | +0.19 | +0.27 | +0.25 |
| 505 | +0.74 | -0.57 | +0.61 | -0.22 | +0.03 | +0.20 |
| 606 | +0.80 | -0.40 | +0.39 | +0.08 | +0.44 | +0.08 |
| 707 | +0.44 | -0.58 | +0.27 | -0.02 | +0.47 | +0.44 |
| 808 | +0.26 | -0.48 | -0.11 | -0.02 | +0.14 | +0.08 |
| median | +0.60 | -0.58 | +0.33 | -0.02 | +0.35 | +0.20 |

The ROF columns reproduce the draft's figure. Under wfull, the contrast reverses: every
Reacher correlation is negative and every LunarLander correlation is positive. Under wdiag,
both tasks are mostly positive and do not separate. Within each Reacher run, the ROF and
wfull series themselves move in opposite directions, so the rise of ROF through Reacher's
training reflects how the latent coordinates rescale, not the reward gradient moving into
observable directions.

Reading: the domain contrast is a native-coordinate measurement and cannot be offered as ROF
behaving as the theory predicts. What remains of the contrast does not use ROF: collapse in
five of eight LunarLander runs against none on Reacher, and the reward-predictability
regressions. Of the geometry results, only the z-only intervention survives a change of
coordinates.

Files: `results/metrics_w_sp<s>_seed*.txt`, `results/metrics_w_r<s>_seed*.txt`; scorer
`results/score_contrast.py`.
