# z-only reward head — seed 12345 continuous (Aug 2026, RTX 5090)

World-model checkpoints for the Section 7 intervention: the reward head is
constrained to read the latent `z` alone, with the recurrent state `h` removed
from its input entirely.

| Field | Value |
|---|---|
| Train seed | 12345 |
| Protocol | Single continuous 500-epoch run (no resume, no optimizer reset) |
| Data | Human `lunarlander_{train,val}_dataset.npz` |
| Config | `config_zonly.yaml` — identical to `config.yaml` plus `world_model.reward_head_input: z` |
| Reward head | `nn.Linear(16, 256)` first layer (`latent_dim` only) vs `nn.Linear(272, 256)` for `[h, z]` |
| GPU | RTX 5090 |
| Checkpoints | 100 files, epochs 5–500, stride 5 |
| MPC outcome | Slow start, mid-run dip, partial rebound. No terminal collapse. |

## Implementation

The ablation is a genuine truncation of the head's input, not a mask with zero
weights. `WorldModel.set_reward_head_input(z_only=True)` replaces
`reward_head[0]` after `__init__` has already built the full head, so every
other module consumes the same RNG draws an ordinary run with this seed would;
the two arms differ in initial weights only at that one layer.

`WorldModel.load_state_dict` infers the variant from the width of
`reward_head.0.weight`, so these checkpoints load in `eval_mpc.py`,
`eval_metrics.py` and `test_policy.py` with no flag and no code path of your own.

## Results

MPC sweep, 20 episodes per checkpoint at eval seed 12345, MA-7 smoothed:

| | z-only 12345 (5090) | control 12345 cont. (5090) | V7 12345 cont. (3080) |
|---|---:|---:|---:|
| MA-7 peak | +143.0 @ ep 305 | +168.0 @ ep 480 | +99.0 @ ep 160 |
| Pre-peak floor | −15.3 @ ep 120 | — | — |
| MA-7 at ep 50 / 100 / 150 | −6.3 / +14.5 / +29.7 | — | — |
| Mid-run valley | +87.2 @ ep 375 (−55.7) | — | — |
| Rebound | +141.9 @ ep 445 | — | — |
| Final-10 raw mean | +124.1 | +157.6 | −31.9 |
| Peak-to-end MA-7 gap | +16.3 | +0.5 | +142.6 |

Two things stand out. The run spends its first ~150 epochs at or below zero and
never reaches the control's ceiling, which is what you would expect if a
meaningful part of the Lander reward is history-dependent and the head can no
longer read `h`. And the mid-run dip recovers, so the endpoint is healthy even
though the trajectory is not monotone.

Validation loss does not see any of this: mean over epochs ≥ 400 is **32.75**
for z-only against **34.68** for the control, i.e. the z-only run looks
slightly *better* by val loss while being clearly worse in closed loop.

## Metrics

`eval_metrics.py` at `n_jac_states=128`, averaged over eval seeds 12345, 1, 2.
Values below are means over epochs > 50.

| | z-only | control |
|---|---:|---:|
| RCF | 0.213 | 0.291 |
| OCF | 0.090 | 0.079 |
| ROF (good pool) | 0.235 | 0.220 |
| ρₛ(ROF, MPC MA-7) | **+0.78** | **−0.26** |
| ρₛ(ROF_combined, MPC MA-7) | **+0.79** | −0.47 (reset arm) |
| good-pool ROF, pre/post-300 | 0.2082 → 0.2689 (**+0.0607**) | — |
| Drawdown gate (good pool) | 0.0070 → **quiet** | 0.0344 → quiet |

The correlation sign flips from negative to positive under the architecture
change, putting Lander in the same regime as Reacher, and ROF rises after epoch
300 rather than falling. RCF drops by about a quarter. OCF shifts too, though it
neither separates collapse from health nor involves the reward head.

The good-pool ROF turning points do not lead MPC here: ROF peaks at epoch 470
while the MPC hole is at 375. Whatever ROF is tracking, it is not an early
warning on this run.

The drawdown gate stays quiet through the dip. Since the run recovers to +124
final-10, quiet is the correct run-level verdict.

## Reproducing

```powershell
# train (note --fresh to avoid the auto-resume trap)
python train_models.py --phase world_model --config config_zonly.yaml --seed 12345 --fresh

# MPC sweep
python eval_mpc.py --checkpoints_dir wm_checkpoints/zonly_12345_5090 `
                   --seed 12345 --episodes 20 --output results/mpc_zonly_12345.txt

# metrics, one eval seed at a time
python eval_metrics.py --config config_zonly.yaml --seed 12345 --compute_dyn_jac false `
                       --n_jac_states 128 `
                       --checkpoints_dir wm_checkpoints/zonly_12345_5090 `
                       --output results/metrics_zonly_seed12345.txt
```

Companion logs:
- MPC sweep: `results/mpc_zonly_12345.txt` (100 ckpts × 20 eps, eval seed 12345)
- Train log: `results/zonly_12345_train.txt` (epochs 1–500, continuous, seed 12345)
- Metrics: `results/metrics_zonly_seed{12345,1,2}.txt` (`n_jac_states=128`)

## Caveat on what this run can and cannot show

Seed 12345 does **not** collapse on the 5090 even with the ordinary `[h, z]`
head — see the control run. So this arm cannot speak to whether the
reward-migration pathway causes collapse: there was no collapse here to
prevent. It establishes the architectural response of the fractions, and the
cost in closed-loop competence of removing `h`, and nothing about mechanism.

The mechanism test is the same intervention on a substrate that does collapse,
i.e. seed 12345 continuous on the 3080, whose `[h, z]` control is
`wm_checkpoints/v7_12345_continuous` (peak +99.0 @ ep 160, final-10 −31.9).
That run is not in this repo yet.
