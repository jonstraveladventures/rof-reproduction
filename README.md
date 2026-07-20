# Reproduction and seed-sensitivity experiments for LunarLander_RSSM

Companion repository to correspondence about arXiv:2607.01736 (*Predicting
Closed-Loop Performance of Latent World Models*). It contains everything
behind the numbers in that correspondence: the modified code with full git
history, the generated datasets, the raw evaluation logs, and the two
companion documents.

This is a fork of `nsmoly/LunarLander_RSSM` at upstream commit `e684ad1`.
All modifications live on the `llc-crossenv` branch;
`git log e684ad1..llc-crossenv` shows exactly what changed and why, commit
by commit. The discrete code path after the modifications is
regression-tested bit-identical to the upstream original, so every
discrete-action result was produced by the upstream pipeline end to end.

## Map

- **Code** (repo root): upstream plus the additions described in the
  commit history. Key added scripts: `collect_dataset_scripted.py`
  (scripted data collector), `wm_mpc_policy_continuous.py` (Gaussian
  CEM-MPC for continuous actions), `analyze_llc.py` (curve
  characterisation, correlation table, composite picks),
  `build_mixtures.py` and `data_measures.py` (mixture/subset datasets),
  `test_theorem1.py` (perturbation probes), `reward_markovianity.py`.
- **Datasets** (repo root, `*.npz`): the author's originals
  (`lunarlander_{train,val}_dataset.npz`), scripted discrete
  (`lunarlander_scripted_*`), scripted continuous
  (`lunarlander_continuous_*`), mixtures (`mix_f{25,50,75}_*`), and
  human subsets (`human_sub{25,50}_train.npz`). Construction recipes and
  seeds are in `docs/repro-guide.pdf`.
- **`results/`**: raw logs for every run reported. Per run: the MPC sweep
  log (per-episode returns for all 100 checkpoints) and three metric-sweep
  files (seeds 12345/1/2). `analysis_*.txt` are the corresponding
  analysis outputs (regime check, correlation table, checkpoint picks).
  File naming: `f1rs` = the seed-777 retrain on the original human data,
  `disc_scripted` = the provenance control, `dq_f25/f75/h25` = the
  mixture and subset cells, `llc`/`c_mpc` = the continuous-action leg.
- **`docs/`**: the two companion documents, with LaTeX sources.
  `repro-guide.pdf` is the step-by-step reproduction guide (the seed-777
  recipe, the checkpoint auto-resume trap, the RNG stream-position
  effect, and all result tables). `fisher-support-note.pdf` is the formal
  note (the observable subspace as Fisher-information support, validity
  probes, and the amortised z-only correction channel).
- **`reacher_rssm/`**: Nikolai's Reacher-v5 port (continuous actions,
  Markovian reward). Negative control for the seed panel: same RSSM +
  metrics + CROF pipeline as lander, self-contained under this
  subfolder. Includes `reacher_{train,val}_dataset.npz`. See
  `reacher_rssm/README.md` for setup and run order. WM checkpoints are
  not committed (same policy as lander WM weights).

## Reproducing the headline

The seed-777 retrain needs only the upstream code: see the reproduction
guide, which also documents the one practical trap (the trainer
auto-resumes from existing checkpoints) and what to expect when comparing
recomputed metrics against logged values.

## Not included

Retrained model checkpoints (100 per run, several GB in total) are not
committed; they are available on request. The `checkpoints_llc_smoke/`
directory referenced by early commits is local-only smoke-test output.
