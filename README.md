# rof-reproduction

Code, datasets and logs for the experiments in arXiv:2607.01736 (version 3 onwards). The paper tests the Reward Observability Fraction (ROF) as an offline checkpoint selector for RSSM world models on Gymnasium LunarLander-v3, with Reacher-v5 as a contrasting task.

The RSSM, planner and training code come from [nsmoly/LunarLander_RSSM](https://github.com/nsmoly/LunarLander_RSSM), a minimal RSSM implementation for LunarLander. This repository is a fork at upstream commit `e684ad1`. All changes are on the `llc-crossenv` branch; `git log e684ad1..llc-crossenv` lists them.

## Contents

**Code** (repository root): upstream code and the scripts for these experiments.

- `eval_metrics.py`: per-checkpoint offline metrics, including ROF on the good and bad validation pools. `--whiten` adds coordinate-invariant variants.
- `analyze_llc.py`: log loading with completeness checks, MA-7 smoothing, the collapse rule and correlation tables.
- `eval_edge.py`: planner-support and planner-advantage probe.
- `eval_closedloop.py`: instrumented closed-loop MPC episodes.
- `test_theorem1.py`: perturbation probes of the Fisher-support statement at the nonlinear model.
- `collect_dataset_scripted.py`, `build_mixtures.py`, `data_measures.py`: scripted data and human/scripted mixtures.
- `wm_mpc_policy_continuous.py`: Gaussian CEM-MPC for the continuous-action variant.

**`reacher_rssm/`**: the Reacher-v5 port (continuous actions), with its own `eval_metrics.py` and datasets. Setup and run order are in `reacher_rssm/README.md`.

**Datasets** (`*.npz`, at the root and in `reacher_rssm/`): human demonstrations (`lunarlander_{train,val}_dataset.npz`), scripted data, mixtures (`mix_f{25,50,75}_*`) and human subsets (`human_sub{25,50}_train.npz`). Recipes and seeds are in `docs/repro-guide.pdf`.

**`results/`**: raw logs for every run.

- `mpc_<run>.txt`: MPC sweeps with per-episode returns at all 100 checkpoints.
- `metrics_<run>_seed{12345,1,2}.txt`: offline metric sweeps, one per metric seed. `metrics_w_<run>_seed*.txt` adds whitened ROF.
- `edge_<run>.txt`, `cl_<run>.txt` and `cl_<run>_raw.npz`: planner probes.
- `gate_calibration.py`, `collapse_detector.py`, `score_whiten.py`, `score_contrast.py` and `figures/`: scripts behind the paper's tables and figures.
- `MANIFEST.txt`: cluster jobs and code commit behind each set of runs.

**`wm_checkpoints/`**: world-model checkpoints (100 per run) for the original run (`12345-3080-R` in the paper, at the top level), `v7_12345_continuous` (`12345-3080-C`), `zonly_12345_3080` (`12345-3080-Z`) and `zonly_12345_5090` (`12345-5090-Z`).

**`docs/`**: results sheets, the Fisher-support note and the reproduction guide, with LaTeX sources.

**`env/`**: package versions used on the cluster and locally.

## Run names

| In `results/` | In the paper |
|---|---|
| `sp101` to `sp808` | eight-seed panel, human data |
| `sp909`, `sp1010`, `f25_*`, `f75_*`, `scr_*`, `h128_*`, `z8_*` | twelve-run panel: training-data mixtures and capacity variants |
| `r101` to `r808` | Reacher eight-seed panel |
| `v7` | `12345-3080-C` |
| `zonly3080`, `zonly` | `12345-3080-Z`, `12345-5090-Z` |
| `dq_f1rs` | `777-L40S-C` |
| `disc_scripted`, `dq_f25`, `dq_f75`, `dq_h25`, `llc`, `c_mpc` | earlier experiments not in the paper (scripted-data control, data-quality cells, continuous actions); see `docs/repro-guide.pdf` |

## Not included

Checkpoints for the panel runs (about 350 MB per run) are not committed and are available on request.
