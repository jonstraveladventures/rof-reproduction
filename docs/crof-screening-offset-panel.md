# LunarLander screening and capacity panel — results (August 2026)

Twelve fresh world-model runs, training seeds 909 and 1010 throughout, trained and evaluated
under the same conventions as the July seed panels. Expected outcomes were written down before
launch. All runs completed cleanly: 500 epochs, 100 checkpoints, MPC on all 100 checkpoints
(20 episodes each, eval seed 12345, L40S GPU), metrics with n_jac_states=128 averaged over
eval seeds 12345/1/2 (ada CPU). Analysis: centred MA-7 smoothing, seed-averaged pools,
epochs > 50 for the drawdown gate, epochs ≥ 50 for level statistics.

## Design

| arm | runs | training data | config |
|---|---|---|---|
| screening | sp909, sp1010 | all-human (750 episodes, standard dataset) | default |
| screening | f75_909, f75_1010 | mix_f75 (562 human + 188 scripted) | default |
| screening | f25_909, f25_1010 | mix_f25 (188 human + 562 scripted) | default |
| screening | scr_909, scr_1010 | all-scripted | default |
| capacity | h128_909, h128_1010 | all-human | hidden_dim 256 → 128 |
| capacity | z8_909, z8_1010 | all-human | latent_dim 16 → 8 |

Each run is evaluated on its own family's validation set (the same convention as the earlier
mixture runs); the capacity variants use the standard validation set.

## Per-run results

mean MPC / peak / final-10 are MA-7 smoothed closed-loop return; collapse is the boolean
(peak minus final-10 exceeding half the smoothed range); drawdown is the good-pool gate
statistic (max drawdown of seed-averaged, MA-7 jac_rof over epochs > 50); bad level is mean
seed-averaged jac_rof_bad over epochs ≥ 50.

| run | mean MPC | peak | final-10 | collapse | drawdown | bad level |
|---|---|---|---|---|---|---|
| sp909 | +104.1 | +160.1 | +119.6 | no | 0.019 | 0.279 |
| sp1010 | +73.5 | +137.5 | +46.5 | yes | 0.009 | 0.237 |
| f75_909 | +88.7 | +151.8 | +8.8 | yes | 0.035 | 0.292 |
| f75_1010 | +86.3 | +136.0 | +64.4 | yes | 0.027 | 0.263 |
| f25_909 | +57.5 | +159.4 | +23.5 | yes | 0.037 | 0.304 |
| f25_1010 | +14.7 | +81.8 | +1.6 | yes | 0.067 | 0.291 |
| scr_909 | −123.6 | −70.3 | −104.7 | no | 0.060 | 0.340 |
| scr_1010 | −181.1 | −100.1 | −224.6 | yes | 0.085 | 0.308 |
| h128_909 | +122.5 | +167.4 | +83.4 | yes | 0.011 | 0.367 |
| h128_1010 | +94.9 | +149.3 | +109.4 | no | 0.029 | 0.347 |
| z8_909 | +57.7 | +114.0 | +30.6 | yes | 0.043 | 0.259 |
| z8_1010 | +110.4 | +151.5 | +116.3 | no | 0.038 | 0.260 |

## Derived numbers

**Spread.** Mean closed-loop return spans −181 to +122. The panel contains never-good runs
(both scr), collapses to near zero (f25_1010, f75_909), moderate collapses, and healthy runs.

**Between-run screening (8 screening runs).** Ranking by bad level: scr_909 (0.340),
scr_1010 (0.308), f25_909 (0.304), f75_909 (0.292), f25_1010 (0.291), sp909 (0.279),
f75_1010 (0.263), sp1010 (0.237). The three scripted-containing runs rank on top and the two
worst closed-loop runs occupy the top two spots. Spearman with mean MPC: bad level −0.64,
val loss −0.50 (val loss is measured on family-matched validation sets, so the cross-family
comparison partly reflects validation-set difficulty).

**Drawdown gate at 0.05, out of sample.** Two true positives (scr_1010 at 0.085, f25_1010 at
0.067); one false alarm (scr_909, a never-good run without a learned-then-lost shape, at
0.060); six boolean-collapses under threshold (sp1010 0.009, h128_909 0.011, f75_1010 0.027,
f75_909 0.035, f25_909 0.037, z8_909 0.043). f75_909 falls from +151.8 to +8.8, as severe in
closed-loop terms as the original paper run, at a drawdown of 0.035. The calibration does not
transfer across data families and does not see moderate, or even some severe, collapses.

**Architecture offsets (capacity arm vs the sp pair, identical data and validation set).**
h128: mean bad level 0.357 vs 0.258 for the sp pair, an offset of +0.098 at comparable
closed-loop quality. z8: 0.259 vs 0.258, an offset of 0.001. Halving the recurrent state
pushes jac_rof_bad up by an amount larger than any data-family offset we have measured;
halving the latent does nothing. Same direction as the z-only intervention (ROF rising when
the head cannot read h).

**Within-family seed spread.** sp909 vs sp1010 differ by 0.042 on bad level with identical
training data, comparable to several between-family gaps, so a cohort baseline for
screening needs several runs, not one.

**Collapse rate.** 3 of the 6 all-human runs collapsed (sp1010, h128_909, z8_909),
consistent with the July panel's 5/8.

## Data

Raw MPC logs (`mpc_<run>.txt`), metric sweeps (`metrics_<run>_seed{12345,1,2}.txt`), and the
analysis scripts will be in the shared repo (`crof-reproduction`, branch `llc-crossenv`)
alongside the July panels. Checkpoints (12 × ~350 MB) are retained and can be pushed on
request.
