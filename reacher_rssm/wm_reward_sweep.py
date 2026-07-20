# ===========================================================================
# Reacher RSSM - WM Reward Sweep (L2a experiment)
#
# For each WorldModel checkpoint:
#   1. Load checkpoint.
#   2. Forward-pass through every val episode (training-mode posterior, mean z).
#   3. Compute WM-predicted-reward val R^2:
#        - overall
#        - terminal-only      (Reacher: just the truncation step per episode)
#        - non-terminal-only  (everything else)
#   4. Log to text file.
#
# After the sweep:
#   * Load the saved L1c MLP val predictions (.npz from reward_predictability_test.py
#     run with --use_next_obs --save_predictions).
#   * Load existing metrics_eval_logs.txt to recover ROF per epoch.
#   * Generate a single 2-panel PNG: WM val R^2 vs epoch + ROF vs epoch, with
#     L1c reference levels as horizontal lines.
#
# Hypothesis (from L1c analysis):
#   * Reacher: WM val R^2 should rise to ~0.9998 (matching L1c) and saturate
#     fast.  ROF stays high.  Boring.
#   * Lander (separate script in repo root): WM val R^2 should climb past the
#     L1c-with-terminals reference of 0.595 because the WM tracks Box2D's
#     awake/contact flags in latent state - something L1c structurally cannot
#     do.  ROF drops in lockstep.
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================

import argparse
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import r2_score

# Local imports (Reacher-specific WM + dataset).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import WorldModel
from train_models import SequenceDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg, log_path=None):
    print(msg, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Chunked-rollout eval.
#
# IMPORTANT (audit fix, 2026-05-17): training uses sequence_length=30 closed-
# loop posterior rollouts. Evaluating each val episode as a single 100-step
# rollout (Reacher) or 1000-step rollout (Lander) is out-of-distribution: the
# GRU was never optimized to maintain state for that long. We split each val
# episode into non-overlapping chunks of length `rollout_horizon` (default 30,
# matching config.yaml's world_model.sequence_length) and bootstrap each
# chunk independently.  Set rollout_horizon to a very large number (e.g.
# 100000) to recover the old single-full-episode behavior for comparison.
# ---------------------------------------------------------------------------
def _build_padded_chunk_batch(val_dataset, rollout_horizon):
    """Split each val episode into non-overlapping chunks of length up to
    rollout_horizon.  The trailing partial chunk per episode is kept (its
    bootstrap-to-prediction distance stays inside the trained distribution).

    Returns padded tensors of shape (C, H_max, ...) where C is the total
    number of chunks across all episodes and H_max <= rollout_horizon is the
    actual max chunk length observed.
    """
    H = int(rollout_horizon)
    chunks = []
    for data_idxs in val_dataset.episode_indices:
        T = len(data_idxs)
        if T == 0:
            continue
        for s in range(0, T, H):
            chunks.append(data_idxs[s:min(s + H, T)])

    C = len(chunks)
    H_per_chunk = np.array([len(c) for c in chunks], dtype=np.int64)
    H_max = int(H_per_chunk.max()) if C > 0 else H
    obs_dim = val_dataset.obs.shape[1]

    obs_pad = np.zeros((C, H_max, obs_dim), dtype=np.float32)
    nxt_pad = np.zeros((C, H_max, obs_dim), dtype=np.float32)
    if val_dataset.actions.ndim == 2:
        action_dim = val_dataset.actions.shape[1]
        act_pad = np.zeros((C, H_max, action_dim), dtype=np.float32)
    else:
        act_pad = np.zeros((C, H_max), dtype=np.int64)
    rew_pad = np.zeros((C, H_max), dtype=np.float32)
    don_pad = np.zeros((C, H_max), dtype=np.int64)
    valid_mask = np.zeros((C, H_max), dtype=np.bool_)
    flat_row_idx = np.concatenate(chunks).astype(np.int64) if chunks else np.array([], dtype=np.int64)

    for i, ci in enumerate(chunks):
        L = len(ci)
        obs_pad[i, :L] = val_dataset.obs[ci]
        nxt_pad[i, :L] = val_dataset.next_obs[ci]
        act_pad[i, :L] = val_dataset.actions[ci]
        rew_pad[i, :L] = val_dataset.rewards[ci]
        don_pad[i, :L] = val_dataset.dones[ci]
        valid_mask[i, :L] = True

    return (obs_pad, nxt_pad, act_pad, rew_pad, don_pad,
            valid_mask, H_per_chunk, flat_row_idx)


@torch.no_grad()
def eval_wm_on_val(world_model, val_dataset, device, action_mode="continuous",
                    rollout_horizon=30, batch_size=512):
    """Forward-pass the WM through every val transition using non-overlapping
    `rollout_horizon`-step chunks (each bootstrapped fresh, exactly matching
    training conditions).

    action_mode: "continuous" for Reacher, "discrete_onehot" for Lander.

    Returns flat arrays aligned with chunk-major dataset rows.
    """
    import torch.nn.functional as F

    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim
    world_model.eval()

    (obs_pad, nxt_pad, act_pad, rew_pad, don_pad,
     valid_mask, H_per_chunk, flat_row_idx) = \
        _build_padded_chunk_batch(val_dataset, rollout_horizon)
    C_total, H_max = valid_mask.shape

    pred_chunks_out = []
    truth_chunks_out = []
    done_chunks_out = []

    for c0 in range(0, C_total, batch_size):
        c1 = min(c0 + batch_size, C_total)
        C = c1 - c0

        obs_t = torch.from_numpy(obs_pad[c0:c1]).to(device)
        nxt_t = torch.from_numpy(nxt_pad[c0:c1]).to(device)
        if action_mode == "discrete_onehot":
            act_t_idx = torch.from_numpy(act_pad[c0:c1]).to(device)
        else:
            act_t = torch.from_numpy(act_pad[c0:c1]).to(device)

        h = rssm.init_hidden(C, device)
        z = torch.zeros(C, latent_dim, device=device)
        a_init = torch.zeros(C, action_dim, device=device)
        h = rssm.update_hidden(h, z, a_init)
        mean_post, _ = rssm.posterior(h, obs_t[:, 0])
        z = mean_post

        pred_pad = torch.zeros(C, H_max, device=device)
        for t in range(H_max):
            if action_mode == "discrete_onehot":
                a_step = F.one_hot(act_t_idx[:, t], num_classes=action_dim).float()
            else:
                a_step = act_t[:, t]
            h = rssm.update_hidden(h, z, a_step)
            mean_post, _ = rssm.posterior(h, nxt_t[:, t])
            z = mean_post
            pred_pad[:, t] = world_model.predict_reward(h, z)

        pred_pad = pred_pad.cpu().numpy()

        for i_local in range(C):
            i = c0 + i_local
            L = int(H_per_chunk[i])
            pred_chunks_out.append(pred_pad[i_local, :L])
            truth_chunks_out.append(rew_pad[i, :L])
            done_chunks_out.append(don_pad[i, :L])

    pred = np.concatenate(pred_chunks_out)
    truth = np.concatenate(truth_chunks_out)
    dones = np.concatenate(done_chunks_out)
    return pred, truth, dones, flat_row_idx


def split_r2(pred, truth, dones):
    """Compute R^2 overall, on terminal rows, and on non-terminal rows."""
    n_term = int((dones == 1).sum())
    n_nonterm = int((dones == 0).sum())
    r2_all = float(r2_score(truth, pred))
    r2_term = float(r2_score(truth[dones == 1], pred[dones == 1])) if n_term >= 2 else float("nan")
    r2_nonterm = float(r2_score(truth[dones == 0], pred[dones == 0])) if n_nonterm >= 2 else float("nan")
    mse_all = float(np.mean((truth - pred) ** 2))
    return r2_all, r2_term, r2_nonterm, mse_all, n_term, n_nonterm


# ---------------------------------------------------------------------------
# ROF parser (lifted style from analyze_metrics.py)
# ---------------------------------------------------------------------------
def parse_rof_per_epoch(path):
    """Return dict {epoch: rof_value} parsed from metrics_eval_logs.txt."""
    if not os.path.isfile(path):
        return {}
    out = {}
    cur_epoch = None
    epoch_re = re.compile(r"=== Epoch (\d+) ===")
    rof_re = re.compile(r"jac_rof=([-+]?\d*\.\d+|\d+|nan)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                cur_epoch = int(m.group(1))
                continue
            if cur_epoch is None:
                continue
            m = rof_re.search(line)
            if m:
                v = m.group(1)
                try:
                    out[cur_epoch] = float(v)
                except ValueError:
                    out[cur_epoch] = float("nan")
    return out


# ---------------------------------------------------------------------------
# L1c reference numbers from saved .npz
# ---------------------------------------------------------------------------
def compute_l1c_reference(npz_path):
    """Load L1c saved val predictions and compute reference R^2 numbers
    (overall, terminal-only, non-terminal-only) on the *same* val set."""
    if not os.path.isfile(npz_path):
        return None
    d = np.load(npz_path)
    pred = d["l1c_pred_val_1step"]
    truth = d["true_val_1step"]
    dones = d["dones_val_1step"]
    n_term = int((dones == 1).sum())
    out = {
        "filter_terminals": bool(d["filter_terminals"]),
        "use_next_obs": bool(d["use_next_obs"]),
        "val_R2_mlp_1step": float(d["val_R2_mlp_1step"]),
        "val_R2_mlp_2step": float(d["val_R2_mlp_2step"]),
        "n_terminal": n_term,
    }
    out["val_R2_split_overall"] = float(r2_score(truth, pred))
    out["val_R2_split_terminal"] = (
        float(r2_score(truth[dones == 1], pred[dones == 1])) if n_term >= 2 else float("nan")
    )
    out["val_R2_split_nonterminal"] = (
        float(r2_score(truth[dones == 0], pred[dones == 0])) if (dones == 0).sum() >= 2 else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(per_epoch_log, l1c_ref, rof_per_epoch, out_png, env_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.array(sorted(per_epoch_log.keys()))
    r2_all = np.array([per_epoch_log[e]["r2_all"] for e in epochs])
    r2_term = np.array([per_epoch_log[e]["r2_term"] for e in epochs])
    r2_nonterm = np.array([per_epoch_log[e]["r2_nonterm"] for e in epochs])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(epochs, r2_all, label="WM val R² (overall)", color="C0", lw=2)
    ax.plot(epochs, r2_nonterm, label="WM val R² (non-terminal)", color="C2", lw=1.5, ls="--")
    if not np.all(np.isnan(r2_term)):
        ax.plot(epochs, r2_term, label="WM val R² (terminal)", color="C3", lw=1.5, ls=":")

    if l1c_ref is not None:
        ax.axhline(l1c_ref["val_R2_split_overall"], color="C0", lw=1.0, ls="--", alpha=0.5,
                   label=f"L1c overall = {l1c_ref['val_R2_split_overall']:.3f}")
        if not np.isnan(l1c_ref["val_R2_split_nonterminal"]):
            ax.axhline(l1c_ref["val_R2_split_nonterminal"], color="C2", lw=1.0, ls=":", alpha=0.5,
                       label=f"L1c non-term = {l1c_ref['val_R2_split_nonterminal']:.3f}")
        if not np.isnan(l1c_ref["val_R2_split_terminal"]):
            ax.axhline(l1c_ref["val_R2_split_terminal"], color="C3", lw=1.0, ls=":", alpha=0.5,
                       label=f"L1c terminal = {l1c_ref['val_R2_split_terminal']:.3f}")

    ax.set_ylabel("WM reward val R²")
    ax.set_title(f"L2a: WM Reward-Head Quality vs Epoch  ({env_label})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(min(-0.1, np.nanmin(r2_all) - 0.05), 1.02)

    ax = axes[1]
    if rof_per_epoch:
        re_x = np.array(sorted(rof_per_epoch.keys()))
        re_y = np.array([rof_per_epoch[e] for e in re_x])
        ax.plot(re_x, re_y, label="ROF (jac_rof)", color="C4", lw=2)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "(no ROF data found)", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("ROF")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"Saved plot to {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reacher WM reward-head sweep (L2a)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--val_dataset", default="reacher_val_dataset.npz")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--metrics_log", default="metrics_eval_logs.txt",
                        help="Existing metrics log file to parse ROF per epoch from.")
    parser.add_argument("--l1c_npz", default="reacher_l1c_predictions.npz",
                        help="L1c predictions saved by reward_predictability_test.py "
                             "(must match val_dataset and be from --use_next_obs config).")
    parser.add_argument("--output", default="wm_reward_sweep_logs.txt")
    parser.add_argument("--out_png", default="wm_reward_vs_epoch.png")
    parser.add_argument("--epoch_min", type=int, default=None)
    parser.add_argument("--epoch_max", type=int, default=None)
    parser.add_argument("--epoch_stride", type=int, default=1)
    parser.add_argument("--single_checkpoint", default=None,
                        help="Path to a single checkpoint - skips the sweep, runs eval, prints.")
    parser.add_argument("--rollout_horizon", type=int, default=30,
                        help="Length of each closed-loop posterior rollout segment, "
                             "matching world_model.sequence_length used at training "
                             "time (default 30).  Each val episode is split into "
                             "non-overlapping chunks of this length and every chunk "
                             "is bootstrapped independently.  Set to a very large "
                             "value (e.g. 100000) to recover the previous "
                             "single-full-episode rollout behavior.")
    parser.add_argument("--chunk_batch_size", type=int, default=512,
                        help="Batch size in number of CHUNKS (not episodes).")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load config (architecture + dataset hyperparams must match training) ---
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    wm_cfg = config.get("world_model", {})
    action_dim = wm_cfg.get("action_dim", 2)
    sequence_length = wm_cfg.get("sequence_length", 30)
    dataset_seq_offset = wm_cfg.get("dataset_seq_offset", 5)
    cap = wm_cfg.get("capacity", {})
    latent_dim = cap.get("latent_dim", 16)
    hidden_dim = cap.get("hidden_dim", 256)
    mlp_hidden_dim = cap.get("mlp_hidden_dim", hidden_dim)
    gru_num_layers = int(cap.get("gru_num_layers", 1))

    # --- Load val dataset ---
    val_path = args.val_dataset if os.path.isabs(args.val_dataset) else \
        os.path.join(script_dir, args.val_dataset)
    val_dataset = SequenceDataset(
        val_path, sequence_length, action_dim,
        random_start=False, dataset_seq_offset=dataset_seq_offset,
    )
    obs_dim = val_dataset.obs_dim
    print(f"Val dataset: {len(val_dataset.episode_indices)} episodes, "
          f"{val_dataset.obs.shape[0]} transitions, obs_dim={obs_dim}")

    # --- Build model shell (weights loaded per checkpoint) ---
    world_model = WorldModel(
        obs_dim, action_dim,
        latent_dim=latent_dim, hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers, mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)

    # --- Single-checkpoint mode (debugging) ---
    if args.single_checkpoint:
        ckpt_path = args.single_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(script_dir, ckpt_path)
        world_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        t0 = time.time()
        pred, truth, dones, _ = eval_wm_on_val(world_model, val_dataset, DEVICE,
                                                action_mode="continuous",
                                                rollout_horizon=args.rollout_horizon,
                                                batch_size=args.chunk_batch_size)
        r2_all, r2_term, r2_nonterm, mse_all, n_term, n_nonterm = split_r2(pred, truth, dones)
        elapsed = time.time() - t0
        print(f"Single-checkpoint eval ({elapsed:.1f}s, rollout_horizon={args.rollout_horizon}):")
        print(f"  N total      = {len(truth)}")
        print(f"  N terminal   = {n_term}")
        print(f"  N non-term   = {n_nonterm}")
        print(f"  WM val R^2 (overall)      = {r2_all:.4f}")
        print(f"  WM val R^2 (terminal)     = {r2_term:.4f}")
        print(f"  WM val R^2 (non-terminal) = {r2_nonterm:.4f}")
        print(f"  WM val MSE                = {mse_all:.6f}")
        return

    # --- Discover checkpoints ---
    ckpt_dir = args.checkpoints_dir if os.path.isabs(args.checkpoints_dir) else \
        os.path.join(script_dir, args.checkpoints_dir)
    checkpoints = []
    for fn in os.listdir(ckpt_dir):
        m = re.search(r"epoch_(\d+)\.pt$", fn)
        if m and fn.startswith("world_model_"):
            checkpoints.append((int(m.group(1)), fn))
    checkpoints.sort()
    if args.epoch_min is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e >= args.epoch_min]
    if args.epoch_max is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e <= args.epoch_max]
    if args.epoch_stride > 1:
        checkpoints = checkpoints[::args.epoch_stride]
    if not checkpoints:
        print(f"No matching checkpoints in {ckpt_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoints (epoch {checkpoints[0][0]} - {checkpoints[-1][0]})")

    # --- Output log header ---
    out_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    log_path = out_path
    open(log_path, "w", encoding="utf-8").close()  # truncate
    log(f"Reacher WM reward-head sweep (L2a)  --  {datetime.now():%Y-%m-%d %H:%M:%S}", log_path)
    log("=" * 78, log_path)
    log(f"val_dataset      = {val_path}", log_path)
    log(f"checkpoints_dir  = {ckpt_dir}", log_path)
    log(f"l1c_npz          = {args.l1c_npz}", log_path)
    log(f"metrics_log      = {args.metrics_log}", log_path)
    log(f"#checkpoints     = {len(checkpoints)}", log_path)
    log(f"#val episodes    = {len(val_dataset.episode_indices)}", log_path)
    log(f"#val transitions = {val_dataset.obs.shape[0]}", log_path)
    log(f"rollout_horizon  = {args.rollout_horizon}  (matches train sequence_length)", log_path)
    log(f"device           = {DEVICE}", log_path)
    log("", log_path)

    # --- L1c reference for the comparison ---
    l1c_npz_path = args.l1c_npz if os.path.isabs(args.l1c_npz) else os.path.join(script_dir, args.l1c_npz)
    l1c_ref = compute_l1c_reference(l1c_npz_path)
    if l1c_ref is None:
        log(f"WARNING: L1c npz not found at {l1c_npz_path}; reference lines disabled", log_path)
    else:
        log("L1c reference (from saved val predictions):", log_path)
        log(f"  filter_terminals = {l1c_ref['filter_terminals']}, "
            f"use_next_obs = {l1c_ref['use_next_obs']}, "
            f"#terminal val rows = {l1c_ref['n_terminal']}", log_path)
        log(f"  L1c val R^2 (overall)      = {l1c_ref['val_R2_split_overall']:.4f}", log_path)
        log(f"  L1c val R^2 (terminal)     = {l1c_ref['val_R2_split_terminal']:.4f}", log_path)
        log(f"  L1c val R^2 (non-terminal) = {l1c_ref['val_R2_split_nonterminal']:.4f}", log_path)
        log("", log_path)

    # --- Run sweep ---
    log("epoch    R2_all     R2_term    R2_nonterm  MSE_all    N_term  N_nonterm  elapsed_s", log_path)
    log("-" * 78, log_path)
    per_epoch = {}
    sweep_t0 = time.time()
    for i, (epoch, fn) in enumerate(checkpoints):
        ckpt_path = os.path.join(ckpt_dir, fn)
        world_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        t0 = time.time()
        pred, truth, dones, _ = eval_wm_on_val(world_model, val_dataset, DEVICE,
                                                action_mode="continuous",
                                                rollout_horizon=args.rollout_horizon,
                                                batch_size=args.chunk_batch_size)
        r2_all, r2_term, r2_nonterm, mse_all, n_term, n_nonterm = split_r2(pred, truth, dones)
        elapsed = time.time() - t0
        per_epoch[epoch] = dict(r2_all=r2_all, r2_term=r2_term, r2_nonterm=r2_nonterm,
                                 mse_all=mse_all, n_term=n_term, n_nonterm=n_nonterm,
                                 elapsed=elapsed)
        log(f"{epoch:>5d}    {r2_all:>+8.4f}  {r2_term:>+8.4f}  {r2_nonterm:>+8.4f}  "
            f"{mse_all:>8.5f}  {n_term:>5d}  {n_nonterm:>9d}  {elapsed:>6.1f}",
            log_path)
    sweep_total = time.time() - sweep_t0
    log("-" * 78, log_path)
    log(f"sweep total wall: {sweep_total:.1f}s ({sweep_total/60:.1f} min)", log_path)

    # --- ROF per-epoch from existing metrics log ---
    metrics_log_path = args.metrics_log if os.path.isabs(args.metrics_log) else \
        os.path.join(script_dir, args.metrics_log)
    rof = parse_rof_per_epoch(metrics_log_path)
    if rof:
        log(f"\nLoaded ROF for {len(rof)} epochs from {metrics_log_path}", log_path)
    else:
        log(f"\nWARNING: no ROF parsed from {metrics_log_path}", log_path)

    # --- Plot ---
    out_png_path = args.out_png if os.path.isabs(args.out_png) else os.path.join(script_dir, args.out_png)
    make_plot(per_epoch, l1c_ref, rof, out_png_path, env_label="Reacher-v5")
    log(f"Saved plot to {out_png_path}", log_path)


if __name__ == "__main__":
    main()
