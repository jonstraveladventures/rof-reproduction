# Reacher RSSM — Design and Technical Spec

This document is the **single source of truth** for the Reacher port. Future
chats / future contributors should read this file first; any decision in
the code that contradicts this file is a bug.

## 1. Purpose

Validate that the Composite Reward–Observability Fraction (CROF) metric
developed for LunarLander_RSSM also predicts closed-loop performance on a
**continuous-action** environment. If yes, the paper claim "CROF works
across discrete and continuous action spaces" is supported by independent
evidence on a standard control benchmark.

We deliberately keep the architecture, metric formulas, and evaluation
protocol as close to the lander pipeline as possible. Only the parts that
fundamentally must change for continuous actions (action input dim, MPC
sampling distribution, A2C policy head) are different.

## 2. Port-out / freeze rule

- `LunarLander_RSSM/` is **frozen** at the version that produced the paper.
  Never edit lander code as part of Reacher work.
- `reacher_rssm/` is **self-contained**: no `import` paths reach out into
  the lander tree. If a metric formula or utility is needed, copy the file
  in.
- When this folder becomes self-sufficient it is copied into a new top-level
  repo `Reacher_RSSM/`. Until then, treat this folder as if it already were
  the new repo: only run scripts from `cd reacher_rssm`, only use folder-
  relative paths.
- `eval_metrics.py` and `analyze_metrics.py` must be **byte-identical** to
  the lander versions (when copied). That is the guarantee that the
  cross-environment CROF comparison is honest. If a real bug is discovered,
  fix it in both repos and re-run lander to verify the headline numbers
  still hold.

## 3. Environment

### Choice and version

- **Gymnasium `Reacher-v5`** (MuJoCo backend). Standard 2-link planar arm,
  fingertip-to-target reach task.
- Episode length is overridden to **`max_episode_steps = 100`** (default is
  50). Rationale: 50 is too short to see the difference between a model that
  reaches and "dwells" at the target vs. one that overshoots; 100 mirrors
  the medium-length setups common in the literature and is closer to the
  600-step lander cap (still much shorter, but Reacher dynamics also reach
  steady state much faster).
- `terminated` is always `False` for Reacher; `done = truncated`. The
  per-step `dones` array in the dataset is therefore zeros for the first 99
  steps and `1` only at step 99.

### Observation, action, reward

| | shape / dtype | meaning |
|---|---|---|
| `obs` | `float32`, `(10,)` | `[cos(θ0), cos(θ1), sin(θ0), sin(θ1), tx, ty, θ̇0, θ̇1, fx-tx, fy-ty]` (Reacher-v5 dropped the always-zero z-component that v4 had) |
| `action` | `float32`, `(2,)`, `Box(-1, 1)` | joint torques on the two revolute joints |
| `reward` | scalar `float32` | `-‖fingertip - target‖ - control_cost`, dense |

Link lengths: `l1 = 0.10`, `l2 = 0.11`. Maximum reach radius `l1 + l2 = 0.21`.
Targets are sampled uniformly inside a disk of radius ≈ 0.20 by the env.

### Hyperparameters

```yaml
env_name: "Reacher-v5"
max_episode_steps: 100
obs_dim: 10
action_dim: 2
action_low: -1.0
action_high:  1.0
link_length_1: 0.10
link_length_2: 0.11
```

## 4. Dataset

### Collection method

The lander dataset was collected by **manual human control**. For Reacher
this is impossible (you cannot meaningfully joystick continuous joint
torques), so we use a programmed mixed-quality controller:

| Bucket | Fraction | Policy |
|---|---|---|
| Good IK | 60% | Analytical 2-link IK target + PD torque controller, small action noise σ = 0.05 |
| Noisy IK | 25% | Same IK + PD, but noise σ = 0.40 (the agent "tries" but executes poorly) |
| Random | 15% | Action ~ Uniform(-1, 1), 2-D, independent per step |

Rationale: matches the lander's empirical mix of clean / partial / failure
trajectories. Noisy-IK was intended to be the "bridge bucket" giving a
smooth good/bad return gradient.

**Empirical outcome (master seed 12345, validated):** σ=0.40 was too high
for the small (l1+l2 = 0.21 m) arm; the noisy-IK bucket collapsed into a
second failure mode with returns ≈ −68 (mean across 188 train episodes,
std ≈ 5.3) rather than the predicted −10..−20 bridge range. The "mid"
curated bucket (`(−25, −10)`) is instead populated naturally by the tail
of good-IK episodes whose IK targets were near the workspace edge or
required large corrections — 188 of the 450 good-IK episodes landed in
the mid bucket from natural action-noise variance, giving us the smooth
gradient we wanted anyway. The metric pipeline only needs populated
good/mid/bad buckets, not specific σ values, so we keep the dataset
as collected. Noisy-IK still contributes value as a *trajectory-style*
distinct from pure random (PD-shaped torques with heavy noise vs.
iid uniform), broadening the world model's training distribution.

### v2 augmentation: constant-action buckets (added after MPC sweep)

The initial 60/25/15 mix produced an accurate-on-paper world model (val
obs-MAE ~0.03, reward-MAE ~0.013, both teacher-forced and open-loop) but
the MPC sweep plateaued at mean return ≈ −30, while an *oracle* MPC that
scored candidate action sequences with the real MuJoCo simulator (same
CEM hyperparameters) reached IK-quality return ≈ −10. The full
diagnostic chain is documented in `wm_mpc_policy_oracle.py` and the
`logs/test_*_e445.txt` files, but the root cause is:

**WM target-attractor bias.** CEM converges to near-zero action samples
(σ ≈ 0.1, μ ≈ 0). Under such sustained near-zero action sequences the
WM's prior drifts the latent toward a "fingertip-at-target" state and
predicts near-zero reward. Reality: zero torque means the arm sits, the
fingertip-to-target distance does not shrink, and you accumulate a
constant distance cost. The dataset contains **zero** sustained
constant-action episodes (IK actions are state-dependent and constantly
changing; the random bucket is iid per-step), so the WM has no training
signal for "what happens when the action is held constant for many
steps."

**Fix (v2):** add two new bucket types in a 3-step pipeline:

| Bucket | Code | Action policy | Train ep. | Val ep. |
|---|---|---|---|---|
| Zero | 3 | `action = (0, 0)` for all 100 steps | 50 | 8 |
| Const | 4 | `action ~ Uniform(-0.4, +0.4)`, sampled once, held for 100 steps | 50 | 8 |

**Why `const_max = 0.4` and not 1.0?** First attempt used `Uniform(-1, +1)`
and the training loss exploded (epoch-1 train_loss ≈ 289 vs v1's ≈ 13). Root
cause: Reacher's shoulder joint has no rotation limit and very low damping,
so sustained `±1` torque spins it up to ±164 rad/s — 4× the max velocity
in any other bucket (random caps at ±38). The recon MSE on those velocity
predictions both wasted WM capacity and was off-distribution from what CEM
actually samples at planning time (CEM `μ` stays near 0 with `σ ~ 0.1–0.5`).
`const_max = 0.4` caps terminal shoulder velocity to ~80 rad/s (≈ 2× random
bucket), keeping the WM in a regime it can already handle while still
teaching "constant input ≠ target attractor." Configurable via the
`--const_max` CLI flag if a different range is needed later.

Validation set is augmented in the same proportions so val MAE during
training stays informative for the new patterns — without that signal
we'd be flying blind on whether retraining actually closed the gap.

Workflow:

```bash
# 1. Collect aux only -> standalone files (no merge yet)
python collect_aux_data.py --headless --seed 999999
#   -> reacher_train_aux.npz   (100 episodes)
#   -> reacher_val_aux.npz     ( 16 episodes)

# 2. Visually verify
python replay_dataset.py --dataset reacher_train_aux.npz --episodes 6
python replay_dataset.py --dataset reacher_train_aux.npz --episodes 3 --class zero
python replay_dataset.py --dataset reacher_train_aux.npz --episodes 3 --class const

# 3. Merge into v1 datasets (re-numbers aux ep_ids; sanity-checks the result)
python merge_datasets.py
#   -> reacher_train_dataset_aug.npz   (750 + 100 = 850 episodes)
#   -> reacher_val_dataset_aug.npz     (122 +  16 = 138 episodes)

# 4. Retrain WM on augmented dataset
python train_models.py \
    --train_dataset reacher_train_dataset_aug.npz \
    --val_dataset   reacher_val_dataset_aug.npz
```

The original 60/25/15 mix is preserved in `reacher_train_dataset.npz`
and `reacher_val_dataset.npz` (never overwritten), so v1 vs v2 ablation
remains possible if needed.

### Sizes

- Training: **750 episodes × 100 steps = 75 000 transitions**.
- Validation: **122 episodes × 100 steps = 12 200 transitions** (16.3% of
  training, matching the lander's val/train ratio).
- Episode counts mirror the lander (750 / 122) for direct comparability of
  per-checkpoint metric statistics. Transition counts are roughly half the
  lander's (lander episodes average ~212 steps; Reacher is fixed at 100).
- Both sets use the same 60 / 25 / 15 bucket mix.
- Approximate real-time cost at MuJoCo `human` render: ~1 s per episode,
  so ~15 min wall clock for the full 872-episode collection.

### Schema (`reacher_train_dataset.npz`, `reacher_val_dataset.npz`)

Identical to the lander schema except `actions` becomes a continuous 2-D
float array. Per-step flat layout (every array has length `N` =
total transitions):

| Key | dtype | shape | notes |
|---|---|---|---|
| `ep_index` | int64 | `(N,)` | 0..(num_episodes-1), groups rows into episodes |
| `step_index` | int64 | `(N,)` | 0..99 within each episode |
| `episode_seed` | int64 | `(N,)` | seed used to `env.reset(seed=...)` for that episode (constant within episode) |
| `obs` | float32 | `(N, 10)` | observation at the current step |
| `actions` | float32 | `(N, 2)` | **continuous**, clipped to [-1, 1] |
| `rewards` | float32 | `(N,)` | per-step reward |
| `next_obs` | float32 | `(N, 10)` | observation at the next step |
| `dones` | int64 | `(N,)` | 0 except 1 at the last step of each episode |
| `episode_class` | int64 | `(N,)` | 0 = good IK, 1 = noisy IK, 2 = random, **3 = zero (v2)**, **4 = const (v2)**; constant within episode |

`episode_class` is a Reacher-specific addition for analysis. Loading code
that mimics the lander's `SequenceDataset` should ignore it if not needed.

### Curated good/bad return thresholds

Per-episode return = sum of per-step rewards over the 100 steps.

```python
R_good = -10.0      # episode_return >= R_good  -> "good" sequence
R_bad  = -25.0      # episode_return <= R_bad   -> "bad" sequence
```

**Empirical return distribution (seed 12345, 750 train + 122 val):**

| Class | n_train | mean | std | range |
|---|---|---|---|---|
| good_ik | 450 | −9.14 | 2.71 | [−19.82, −2.79] |
| noisy_ik | 188 | −68.29 | 5.31 | [−83.02, −53.18] |
| random | 112 | −84.47 | 5.51 | [−97.14, −71.39] |

**Curated bucket counts (train): good=262, mid=188, bad=300.** The mid
bucket is populated by good-IK tail episodes (see Collection method,
above). Val mirrors this proportionally: good=46, mid=27, bad=49.

Thresholds keep the same shape as lander's ±100 cuts; no retuning was
needed.

### Determinism

- Master RNG seed (CLI `--seed`, default `12345`) drives:
  - Per-episode `episode_seed` for `env.reset(seed=...)`.
  - The episode-class assignment ordering (good / noisy / random).
  - The action-noise streams.
- A single seeded run produces a bit-for-bit identical dataset. (Modulo
  any nondeterminism in MuJoCo itself, which in practice is reproducible
  on the same machine for these tiny systems.)

## 5. World model architecture (planned port from lander)

Same RSSM core as lander, with the **only** changes being:

1. **Action input dim** changes from 4 (one-hot discrete) to 2 (continuous,
   no encoding). The transition GRU input becomes `[z; u]` with `u ∈ ℝ²`.
2. **Decoder**: drop the contact head and the done head (Reacher has
   neither contacts nor variable-length terminations). Decoder becomes a
   single MSE-regression head producing `obs ∈ ℝ¹¹`. Reward head stays
   identical (separate 3-layer MLP on `[h; z]`).
3. **Loss weights**: drop `w_d` (no done loss). Keep `w_r = 1.0`,
   `w_ρ = 1.2`, `w_KL = 1.0`, `β_KL = 0.5`. May need to revisit `w_ρ`
   given Reacher's small reward magnitudes.
4. **Latent dims**: keep `d_h = 256`, `d_z = 16` for direct comparability
   with lander (full latent state `s ∈ ℝ²⁷²`). May reduce later if
   underused.
5. **Encoder**: 2-layer MLP, input dim 10 (was 8). Otherwise identical.

Sequence length, stride, KL free-bits, optimizer (AdamW 1e-4), batch size
(64), and total epochs (500) are all **copied unchanged** from lander
config.

## 6. MPC (planned: Gaussian CEM, continuous-action analogue)

Discrete CEM (lander) → Gaussian CEM (Reacher). At each real timestep:

1. Maintain a per-step diagonal Gaussian distribution
   `N(μ_k, σ_k²)` over the 2-D action for `k = 0..H-1`. Initialize
   `μ = 0`, `σ = 0.5`. Floor `σ ≥ σ_min = 0.05` to prevent collapse.
2. Sample `N = 384` candidate action sequences and clip to `[-1, 1]`.
3. Roll each candidate through the world-model prior open-loop, accumulate
   discounted reward over `H = 25` steps with `γ = 0.97`.
4. Take the top `E = 48` elites. Update `μ_k, σ_k` to the elite mean and
   elite std with smoothing `α = 0.7`:
   `μ_k ← α · elite_mean + (1-α) · μ_k`,
   `σ_k ← α · elite_std + (1-α) · σ_k`.
5. Repeat for `I = 4` iterations. Execute the first action of the best
   sequence and re-plan at the next timestep.

Hyperparameters `H, N, E, I, α, γ` are kept identical to lander to
maximize comparability of the per-checkpoint MPC sweep. The only change is
the sampling distribution (categorical → Gaussian).

## 7. Actor-critic (planned: Gaussian policy via latent imagination)

- Actor: 3-layer MLP (hidden 128) reading `[h; z] ∈ ℝ²⁷²`, outputs
  `μ ∈ ℝ²` and `log σ ∈ ℝ²`. Action sample `a ~ tanh(N(μ, σ²))` (squashed
  Gaussian to respect the `[-1, 1]` action bounds). Log-prob uses the
  standard tanh-squashed Gaussian Jacobian correction.
- Critic: 3-layer MLP (hidden 128), scalar value output. Identical to
  lander.
- Training: `1000` epochs, `λ`-returns (`γ = 0.99`, `λ = 0.95`), batch
  size `256`, lr `3e-5`, entropy coefficient decayed from `0.5` to `0.15`.
  All identical to lander.
- Imagination horizon: `15` steps after a `5`-step posterior warm-up.
  Identical to lander.

Model-free baseline: same continuous A2C (or PPO if A2C is too unstable on
Reacher), trained directly in the real environment. Hyperparameters TBD.

## 8. Metrics suite (verbatim copy from lander)

The 40-metric suite is environment-agnostic and **must not be modified**
during the port. To be copied byte-for-byte from
`LunarLander_RSSM/eval_metrics.py`. Things that need parameter changes
only (not formula changes):

- `obs_dim = 10` (was 8)
- `action_dim = 2` (was 4)
- Drop the contact-related logic (no `contact head` to evaluate)
- Drop the done-related logic (no `done head` to evaluate)
- Good/bad return thresholds: `R_good = -10`, `R_bad = -25` (was ±100)

The four subspace-alignment fractions and CROF are unchanged; the
controllability matrix is `B ∈ ℝⁿˣ²` instead of `ℝⁿˣ⁴` but the formula
machinery is the same (singular-value decomposition, effective-rank
threshold `1e-3`, etc.).

### Metric formulas (paper §3.4–§3.5, reproduced for cross-reference)

For the linearization at a sampled latent state `s = [h; z]`:

- `A = ∂f/∂s ∈ ℝⁿˣⁿ`, `B = ∂f/∂u ∈ ℝⁿˣ²`,
  `C = ∂g/∂s ∈ ℝ¹¹ˣⁿ`, `R = ∂ρ/∂s ∈ ℝ¹ˣⁿ`.
- Controllability matrix (fixed linearization):
  `C_H = [B, AB, A²B, ..., A^(H-1) B]`.
- Observability matrix (fixed linearization):
  `O_H = [C; CA; CA²; ...; CA^(H-1)]`.
- SVD: `C_H = Uc Σc Vc^T`, `O_H = Uo Σo Vo^T`.
- Effective ranks `kc, ko` use threshold `σ_i / σ_max ≥ 1e-3`.
- Subspace bases: `Uc^(k) ∈ ℝⁿˣᵏᶜ`, `Vo^(k) ∈ ℝⁿˣᵏᵒ`.
- Reward gradient: `r = Rᵀ ∈ ℝⁿ`.

Subspace-alignment scores (all in `[0, 1]`, unchanged from paper):

```
RCF       = ‖Uc^(k)ᵀ r‖² / ‖r‖²              (reward in controllable subspace)
OCF       = ‖C · Uc^(k)‖_F² / ‖C‖_F²          (decoder sensitivity in ctrl subspace)
ROF       = ‖Vo^(k)ᵀ r‖² / ‖r‖²              (reward in observable subspace; LOWER is better)
CO_overlap = ‖Uc^(k)ᵀ Vo^(k)‖_F² / min(kc, ko)
```

Curated good/bad ROF (Reacher uses the same combiner as lander):

```
jac_rof_combined = 0.5 · jac_rof + 0.5 · jac_rof_bad
```

with `jac_rof` evaluated on N_J = 128 "good" states (return ≥ -10) and
`jac_rof_bad` on N_J = 128 "bad" states (return ≤ -25).

### CROF (paper §3.5, equations 6–7, unchanged)

For each metric `m`, let `m̃` denote min–max normalization across the
training-sweep checkpoints. Then:

```
CROF-A = ROF̃ + 1.0·(1 - k̃c) + 1.0·(1 - k̃o) + 1.0·ẽ_obs
CROF-B = ROF̃ + 0.5·(1 - k̃c) + 0.5·(1 - k̃o) + 0.5·ẽ_obs
```

where `ROF = jac_rof_combined`, `kc = jac_ctrl_rank`, `ko = jac_obs_rank`,
`e_obs = ol_obs_avg`. **Lower CROF = better checkpoint.** All four
normalized terms are oriented so smaller is better.

## 9. Evaluation protocol (planned: identical to lander)

- **MPC sweep**: every world-model checkpoint (every 5 epochs of training,
  100 checkpoints total) is evaluated for 20 episodes with deterministic
  seed sequence `12345 + k` for `k = 0..19`. Per-checkpoint mean return is
  the raw signal; we use a 7-point moving average (MA-7) over the
  100-checkpoint sweep as the smoothed oracle.
- **A2C eval**: each saved A2C checkpoint is run for 100 deterministic
  episodes (`12345 + k`, `k = 0..99`), at episode length 100. Reported:
  mean return, worst-case return, perfect-reach count, catastrophic-fail
  count.
- **Catastrophic threshold**: TBD for Reacher (lander used return < -100;
  for Reacher the analogue might be return < -25 or unreached-target
  fraction). Will be pinned once we see the eval distribution.
- **Perfect threshold**: TBD for Reacher. Likely "fingertip within 0.02 of
  target for the last 20 steps" or "episode return ≥ -3".

## 10. Open questions / TBD

These are intentionally not pinned yet; we will resolve them as data comes
in.

1. **Catastrophic / perfect thresholds for Reacher A2C eval** (see §9).
2. **Reward-loss weight `w_ρ`** — Reacher rewards are roughly 100× smaller
   in magnitude than lander rewards (range ≈ -0.5..-0.001 vs ±100). May
   need to bump `w_ρ` to keep the reward head from being learned too
   slowly relative to obs reconstruction.
3. **Latent dim `d_z`** — 16 may be over-parameterized for a 10-D obs
   space. We will keep 16 for first comparability run, then ablate if
   results suggest it is needed.
4. **Time-varying Jacobian `--compute_dyn_jac`** — keep available but
   default off, like lander. Lander empirically found fixed wins; Reacher
   may differ but not worth pre-tuning.
5. **Model-free baseline** — A2C from lander or switch to PPO/SAC for
   continuous? Decide after first WM sweep is done so we know the bar to
   beat.
