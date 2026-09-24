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
