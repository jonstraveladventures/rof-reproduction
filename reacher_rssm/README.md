# Reacher RSSM (negative control)

Self-contained port of the LunarLander CROF / RSSM pipeline to Gymnasium
**Reacher-v5** (MuJoCo, 10-D observation, 2-D continuous torques). Intended
role in `crof-reproduction`: **Markovian-reward negative control** for the
seed-panel and cross-environment CROF comparison (see parent `docs/repro-guide`
and the LunarLander paper Section 4.8).

No imports from the parent LunarLander code at repo root. Jacobian / CROF
formulas in `eval_metrics.py` and `analyze_metrics.py` match the lander
implementations (same metric definitions; env-specific dims and return
thresholds only).

Full technical spec: [`DESIGN.md`](DESIGN.md).

## Status

| Component | Status |
|-----------|--------|
| Dataset collection (`collect_dataset.py`, `ik_controller.py`) | Done |
| World-model training (`train_models.py --phase world_model`) | Done |
| Gaussian-CEM MPC (`wm_mpc_policy.py`, `eval_mpc.py`) | Done |
| Metrics + CROF (`eval_metrics.py`, `analyze_metrics.py`) | Done |
| Reward predictability (L1 / L2a) | Done |
| Actor-critic (continuous Gaussian policy) | Not ported |

## Setup

From this folder:

```powershell
cd reacher_rssm
pip install -r requirements.txt
# PyTorch: install separately for your CUDA/CPU stack
pip install torch
```

MuJoCo is bundled with the `mujoco` wheel; no separate system install needed.

## What is committed vs local-only

**Committed:** `reacher_train_dataset.npz`, `reacher_val_dataset.npz`
(~19 MB + ~3 MB; v2 IK/mixed-policy recipe, see `DESIGN.md`).

**Local only** (per `.gitignore`): `*.pt`, `checkpoints/`, `logs/`, `.venv/`.
Generate WM checkpoints locally (or transfer on request, same as lander).

## Run order (seed-panel cell)

All commands run **from `reacher_rssm/`** unless noted.

```powershell
# 1. Datasets are already in this folder (reacher_train/val_dataset.npz).
#    To regenerate instead:
# python collect_dataset.py --headless --train_episodes 1500 --val_episodes 240 --seed 12345

# 2. Train world model (single continuous run; use --fresh + dedicated ckpt dir)
python train_models.py --phase world_model --config config.yaml --seed 777 --fresh `
  --checkpoint_dir checkpoints_seed777

# 3. MPC sweep (20 eps/checkpoint, eval seed 12345 — same protocol as lander)
python eval_mpc.py --checkpoints_dir checkpoints_seed777 --seed 12345 --episodes 20 `
  --output ../results/reacher_mpc_seed777.txt

# 4. Metrics sweep (one or three metric seeds, as for lander)
python eval_metrics.py --config config.yaml --seed 12345 --compute_dyn_jac false `
  --checkpoints_dir checkpoints_seed777 `
  --output ../results/reacher_metrics_seed777_seed12345.txt

# 5. Analysis (from repo root; parent analyze_llc.py)
cd ..
python analyze_llc.py --mpc_log results/reacher_mpc_seed777.txt `
  --metrics_logs "results/reacher_metrics_seed777_seed*.txt"
```

Expected qualitative behaviour (paper / repro-guide): reward is Markovian in
`(o_t, a_t)` (MLP R² ≈ 0.9998); MPC return rises near-monotonically; no
late-training collapse; `jac_rof_combined` is weakly correlated with MPC
(opposite sign to LunarLander's migration-collapse run).

## Layout (committed files)

```
reacher_rssm/
  README.md, DESIGN.md, config.yaml, requirements.txt, .gitignore
  collect_dataset.py, ik_controller.py, replay_dataset.py
  models.py, train_models.py, test_worldmodel.py
  wm_mpc_policy.py, eval_mpc.py, diagnose_mpc.py
  eval_metrics.py, analyze_metrics.py
  reward_predictability_test.py, wm_reward_sweep.py   # optional L1/L2a diagnostics
  checkpoints_seed*/     <- created at train time (not committed)
  reacher_train_dataset.npz, reacher_val_dataset.npz  <- committed
```

## Relation to LunarLander_RSSM

Developed under `LunarLander_RSSM/reacher_rssm/` (gitignored there to keep
the paper repo frozen). This copy is the check-in target for
`jonstraveladventures/crof-reproduction`. Do not modify lander code when
extending Reacher.
