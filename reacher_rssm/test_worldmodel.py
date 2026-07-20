# ===========================================================================
# Reacher RSSM - World-model offline tests (ported from lander test_worldmodel.py).
#
# Three modes:
#   teacher  - Posterior teacher-forced rollout. At every step use the real
#              obs to update the posterior, decode (h, z) to predicted
#              next-obs + reward. Compare to GT. This is the WM's "best case".
#
#   open     - Bootstrap from obs[0] once, then unroll the PRIOR for the
#              rest of the episode using the dataset's ground-truth actions.
#              Compare to GT. This is the MPC scenario: if the prior cannot
#              track a real trajectory when handed the correct actions, MPC
#              cannot work no matter how it is tuned.
#
#   symmetry - For Reacher, "control-sensitivity" test. From teacher-forced
#              warmed-up states, apply each of 5 canonical actions for one
#              step and measure mean delta obs per action. Sanity-checks
#              physical plausibility: +tau_0 and -tau_0 should produce
#              opposite delta(dtheta_0); larger torque magnitude should
#              produce larger angular velocity change; etc.
#
# Differences from the lander script:
#   - Continuous 2-D float actions (no one-hot).
#   - 10-D Reacher obs labels.
#   - No done head / done predictions (Reacher episodes are fixed-length).
#   - sim mode and matplotlib animation dropped.
#   - Output: --log_path to tee stdout to a file; plots saved under --plot_dir.
#
# CLI examples:
#   python test_worldmodel.py --mode teacher  --checkpoint checkpoints/world_model_..._epoch_445.pt
#   python test_worldmodel.py --mode open     --checkpoint checkpoints/world_model_..._epoch_445.pt
#   python test_worldmodel.py --mode symmetry --checkpoint checkpoints/world_model_..._epoch_445.pt
# ===========================================================================

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import yaml

from models import WorldModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reacher-v5 obs layout (see DESIGN.md section 3).
OBS_NAMES = [
    "cos(th0)", "cos(th1)",         # 0, 1
    "sin(th0)", "sin(th1)",         # 2, 3
    "target_x", "target_y",         # 4, 5
    "dth0/dt", "dth1/dt",           # 6, 7
    "dx_offset", "dy_offset",       # 8, 9 -- drives reward
]


# ---------------------------------------------------------------------------
# Tee logger: print to console AND append to a log file (for easy copy/paste).
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, log_path: Optional[str]):
        self.log_path = log_path
        self._f = None
        if log_path:
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._f = open(log_path, "w", encoding="utf-8")

    def __call__(self, msg: str = ""):
        print(msg)
        if self._f is not None:
            self._f.write(msg + "\n")
            self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------------------
# Config + checkpoint loading (matches the lander script).
# ---------------------------------------------------------------------------
def load_config(config_path: str):
    if not os.path.exists(config_path):
        return 16, 256, 256, 1, 2
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm = config.get("world_model", {})
    cap = wm.get("capacity", {})
    return (
        int(cap.get("latent_dim",     16)),
        int(cap.get("hidden_dim",     256)),
        int(cap.get("mlp_hidden_dim", cap.get("hidden_dim", 256))),
        int(cap.get("gru_num_layers", 1)),
        int(wm.get("action_dim",      2)),
    )


def load_checkpoint(world_model, checkpoint_path: str, log):
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        world_model.load_state_dict(state_dict)
        log(f"Loaded world model checkpoint: {checkpoint_path}")
        return
    if os.path.exists("world_model.pt"):
        state = torch.load("world_model.pt", map_location=DEVICE, weights_only=True)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        world_model.load_state_dict(state_dict)
        log("Loaded world model from world_model.pt")
        return
    raise FileNotFoundError("No world model checkpoint found.")


# ---------------------------------------------------------------------------
# Dataset loading -- group rows by episode and sort by step_index.
# ---------------------------------------------------------------------------
def load_episodes(dataset_path: str):
    data = np.load(dataset_path)
    obs        = data["obs"].astype(np.float32)
    actions    = data["actions"].astype(np.float32)         # continuous (N, 2)
    rewards    = data["rewards"].astype(np.float32)
    next_obs   = data["next_obs"].astype(np.float32)
    dones      = data["dones"].astype(np.int64)
    ep_index   = data["ep_index"].astype(np.int64)
    step_index = data["step_index"].astype(np.int64)

    episodes = []
    for ep_id in np.unique(ep_index):
        idxs = np.where(ep_index == ep_id)[0]
        if idxs.size == 0:
            continue
        order = np.argsort(step_index[idxs], kind="stable")
        idxs = idxs[order]
        episodes.append({
            "ep_id":    int(ep_id),
            "obs":      obs[idxs],
            "next_obs": next_obs[idxs],
            "actions":  actions[idxs],
            "rewards":  rewards[idxs],
            "dones":    dones[idxs],
        })
    return episodes


# ---------------------------------------------------------------------------
# Plot a single episode: GT vs predicted, one subplot per obs dim + reward.
# ---------------------------------------------------------------------------
def plot_observations(obs_gt, obs_pred, title, rewards_gt=None, reward_pred=None,
                      plot_dir=None, filename="plot.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")   # avoid Tk display issues when running headless
        import matplotlib.pyplot as plt
    except Exception:
        print("Plotting requested but matplotlib is not available.")
        return None

    obs_dim = obs_gt.shape[1]
    has_reward = rewards_gt is not None and reward_pred is not None
    n_rows = obs_dim + (1 if has_reward else 0)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for d in range(obs_dim):
        axes[d].plot(obs_gt[:, d], label="gt", color="black", linewidth=1.5)
        axes[d].plot(obs_pred[:, d], label="pred", color="tab:red", alpha=0.8, linewidth=1.2)
        label = OBS_NAMES[d] if d < len(OBS_NAMES) else f"obs[{d}]"
        axes[d].set_ylabel(label, fontsize=9)
        axes[d].grid(True, alpha=0.3)
        if d == 0:
            axes[d].legend(loc="upper right", fontsize=8)

    if has_reward:
        n = min(len(rewards_gt), len(reward_pred))
        axes[obs_dim].plot(rewards_gt[:n], label="gt", color="black", linewidth=1.5)
        axes[obs_dim].plot(reward_pred[:n], label="pred", color="tab:red", alpha=0.8, linewidth=1.2)
        axes[obs_dim].set_ylabel("reward", fontsize=9)
        axes[obs_dim].grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, filename)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    else:
        plt.show()
        return None


# ---------------------------------------------------------------------------
# Teacher-forced rollout.
# At every step, posterior is conditioned on the real next_obs; we record
# the predicted reconstruction + reward at each step and compare to GT.
# Convention matches training: (h_{t+1}, z_{t+1}) predicts next_obs[t] and rewards[t].
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_teacher(world_model, episode):
    obs        = episode["obs"]            # s_0..s_{N-1}
    next_obs   = episode["next_obs"]       # s_1..s_N
    actions    = episode["actions"]        # a_0..a_{N-1}
    rewards_gt = episode["rewards"]
    dones      = episode["dones"]

    obs_pred:    list = []
    reward_pred: list = []

    h = world_model.rssm.init_hidden(1, DEVICE)
    z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
    a_init = torch.zeros(1, world_model.rssm.action_dim, device=DEVICE)

    # Bootstrap: condition posterior on obs[0].
    h = world_model.rssm.update_hidden(h, z, a_init)
    obs_t = torch.tensor(obs[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mean_post, _ = world_model.rssm.posterior(h, obs_t)
    z = mean_post

    for t in range(len(actions)):
        a_t = torch.tensor(actions[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        h = world_model.rssm.update_hidden(h, z, a_t)
        obs_next_real = torch.tensor(next_obs[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean_post, _ = world_model.rssm.posterior(h, obs_next_real)
        z = mean_post

        obs_pred.append(world_model.reconstruct_obs(h, z).squeeze(0).cpu().numpy())
        reward_pred.append(float(world_model.predict_reward(h, z).item()))

        if dones[t]:
            break

    obs_pred_a    = np.asarray(obs_pred,    dtype=np.float32)
    reward_pred_a = np.asarray(reward_pred, dtype=np.float32)
    obs_gt_c      = next_obs[: len(obs_pred_a)]
    rewards_gt_c  = rewards_gt[: len(reward_pred_a)]
    return obs_gt_c, obs_pred_a, rewards_gt_c, reward_pred_a


# ---------------------------------------------------------------------------
# Open-loop rollout. Bootstrap from obs[0], then prior only.
# THIS is the MPC scenario.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_open(world_model, episode):
    obs        = episode["obs"]
    next_obs   = episode["next_obs"]
    actions    = episode["actions"]
    rewards_gt = episode["rewards"]
    dones      = episode["dones"]

    obs_pred:    list = []
    reward_pred: list = []

    h = world_model.rssm.init_hidden(1, DEVICE)
    z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
    a_init = torch.zeros(1, world_model.rssm.action_dim, device=DEVICE)

    h = world_model.rssm.update_hidden(h, z, a_init)
    obs_t = torch.tensor(obs[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mean_post, _ = world_model.rssm.posterior(h, obs_t)
    z = mean_post

    for t in range(len(actions)):
        a_t = torch.tensor(actions[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        h = world_model.rssm.update_hidden(h, z, a_t)
        mean_prior, _ = world_model.rssm.prior(h)
        z = mean_prior

        obs_pred.append(world_model.reconstruct_obs(h, z).squeeze(0).cpu().numpy())
        reward_pred.append(float(world_model.predict_reward(h, z).item()))

        if dones[t]:
            break

    obs_pred_a    = np.asarray(obs_pred,    dtype=np.float32)
    reward_pred_a = np.asarray(reward_pred, dtype=np.float32)
    obs_gt_c      = next_obs[: len(obs_pred_a)]
    rewards_gt_c  = rewards_gt[: len(reward_pred_a)]
    return obs_gt_c, obs_pred_a, rewards_gt_c, reward_pred_a


# ---------------------------------------------------------------------------
# Metrics helpers.
# ---------------------------------------------------------------------------
def compute_obs_metrics(obs_gt, obs_pred):
    if obs_gt.size == 0 or obs_pred.size == 0:
        return None
    n = min(len(obs_gt), len(obs_pred))
    gt   = obs_gt[:n]
    pred = obs_pred[:n]
    mae  = float(np.mean(np.abs(gt - pred)))
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    # Per-dim MAE so we can see which obs are predicted well.
    per_dim_mae = np.mean(np.abs(gt - pred), axis=0)
    # sin/cos unit-circle deviation.
    norm_sh = np.sqrt(pred[:, 0] ** 2 + pred[:, 2] ** 2)
    norm_el = np.sqrt(pred[:, 1] ** 2 + pred[:, 3] ** 2)
    sincos_mae = float(np.mean(np.maximum(np.abs(norm_sh - 1.0), np.abs(norm_el - 1.0))))
    return {
        "mae":         mae,
        "rmse":        rmse,
        "per_dim_mae": per_dim_mae,
        "sincos_mae":  sincos_mae,
    }


def compute_reward_metrics(rewards_gt, reward_pred):
    if len(rewards_gt) == 0 or len(reward_pred) == 0:
        return None
    n = min(len(rewards_gt), len(reward_pred))
    gt   = rewards_gt[:n]
    pred = reward_pred[:n]
    return {
        "mae":           float(np.mean(np.abs(gt - pred))),
        "rmse":          float(np.sqrt(np.mean((gt - pred) ** 2))),
        "return_gt":     float(np.sum(gt)),
        "return_pred":   float(np.sum(pred)),
    }


def format_obs_metrics(o):
    parts = [
        f"obs MAE={o['mae']:.4f}",
        f"RMSE={o['rmse']:.4f}",
        f"sin/cos MAE={o['sincos_mae']:.4f}",
    ]
    return " ".join(parts)


def format_per_dim(o):
    parts = []
    for d in range(min(len(OBS_NAMES), len(o['per_dim_mae']))):
        parts.append(f"{OBS_NAMES[d]}={o['per_dim_mae'][d]:.4f}")
    return "  per-dim MAE: " + ", ".join(parts)


def format_reward_metrics(r):
    return (f"reward MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}  "
            f"return gt={r['return_gt']:+.2f} pred={r['return_pred']:+.2f}")


# ---------------------------------------------------------------------------
# Control-sensitivity test (Reacher version of the lander's symmetry test).
# ---------------------------------------------------------------------------
PROBE_ACTIONS = [
    ("zero",      np.array([ 0.0,  0.0], dtype=np.float32)),
    ("+tau_0",    np.array([+1.0,  0.0], dtype=np.float32)),
    ("-tau_0",    np.array([-1.0,  0.0], dtype=np.float32)),
    ("+tau_1",    np.array([ 0.0, +1.0], dtype=np.float32)),
    ("-tau_1",    np.array([ 0.0, -1.0], dtype=np.float32)),
]


@torch.no_grad()
def control_sensitivity_test(world_model, episodes, log, num_episodes=10,
                             warmup_steps=20, sample_every=5, seed=0):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(episodes),
                       size=min(num_episodes, len(episodes)), replace=False)

    action_dim = world_model.rssm.action_dim
    latent_dim = world_model.rssm.latent_dim

    states = []   # list of (h, z, obs_recon) tuples to probe from
    for ep_idx in picks:
        ep        = episodes[ep_idx]
        obs       = ep["obs"]
        next_obs  = ep["next_obs"]
        actions   = ep["actions"]
        max_t     = min(len(actions), warmup_steps + sample_every * 10)

        h = world_model.rssm.init_hidden(1, DEVICE)
        z = torch.zeros(1, latent_dim, device=DEVICE)
        a_init = torch.zeros(1, action_dim, device=DEVICE)
        h = world_model.rssm.update_hidden(h, z, a_init)
        obs_t = torch.tensor(obs[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean_post, _ = world_model.rssm.posterior(h, obs_t)
        z = mean_post

        for t in range(max_t):
            a_t = torch.tensor(actions[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            h = world_model.rssm.update_hidden(h, z, a_t)
            obs_next_t = torch.tensor(next_obs[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean_post, _ = world_model.rssm.posterior(h, obs_next_t)
            z = mean_post
            if t >= warmup_steps and (t - warmup_steps) % sample_every == 0:
                obs_now = world_model.reconstruct_obs(h, z).squeeze(0).cpu().numpy()
                states.append((h.clone(), z.clone(), obs_now))

    if not states:
        log("No states collected for symmetry test.")
        return

    log("")
    log(f"Control-sensitivity test: {len(states)} states from {len(picks)} episodes")
    log(f"(warmup={warmup_steps}, sample_every={sample_every})")
    log("")

    # For each canonical probe action, accumulate the delta obs produced by
    # one prior-step from each warmed state.
    deltas = {name: [] for name, _ in PROBE_ACTIONS}
    for h_state, z_state, obs_before in states:
        for name, action in PROBE_ACTIONS:
            a = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            h_next = world_model.rssm.update_hidden(h_state, z_state, a)
            mean_prior, _ = world_model.rssm.prior(h_next)
            obs_after = world_model.reconstruct_obs(h_next, mean_prior).squeeze(0).cpu().numpy()
            deltas[name].append(obs_after - obs_before)

    mean_deltas = {name: np.mean(np.stack(deltas[name], axis=0), axis=0) for name, _ in PROBE_ACTIONS}
    std_deltas  = {name: np.std (np.stack(deltas[name], axis=0), axis=0) for name, _ in PROBE_ACTIONS}

    n_obs = min(len(OBS_NAMES), len(mean_deltas["zero"]))

    # Mean delta-obs per probe action.
    log("Mean Delta-obs per probe action:")
    header = f"{'dim':<12}" + "".join(f"{name:>12}" for name, _ in PROBE_ACTIONS)
    log(header)
    log("-" * len(header))
    for d in range(n_obs):
        row = f"{OBS_NAMES[d]:<12}"
        for name, _ in PROBE_ACTIONS:
            row += f"{mean_deltas[name][d]:>+12.5f}"
        log(row)
    log("")
    log("Std of Delta-obs across probed states:")
    log(header)
    log("-" * len(header))
    for d in range(n_obs):
        row = f"{OBS_NAMES[d]:<12}"
        for name, _ in PROBE_ACTIONS:
            row += f"{std_deltas[name][d]:>12.5f}"
        log(row)
    log("")

    # Sanity checks specific to Reacher.
    log("=== Reacher physics sanity checks (relative to 'zero' baseline) ===")
    z_base = mean_deltas["zero"]
    rel = {name: mean_deltas[name] - z_base for name, _ in PROBE_ACTIONS}

    # dth0/dt is obs index 6; dth1/dt is obs index 7.
    d_w0_pos = rel["+tau_0"][6]; d_w0_neg = rel["-tau_0"][6]
    d_w1_pos = rel["+tau_1"][7]; d_w1_neg = rel["-tau_1"][7]

    def ratio_str(a, b):
        if abs(b) < 1e-8:
            return "inf"
        return f"{abs(a / b):.3f}"

    log(f"Delta(dth0/dt)   +tau_0={d_w0_pos:+.5f}  -tau_0={d_w0_neg:+.5f}  "
        f"|ratio|={ratio_str(d_w0_pos, d_w0_neg)}  "
        f"signs={'opposite OK' if d_w0_pos * d_w0_neg < 0 else 'SAME -- BAD'}")
    log(f"Delta(dth1/dt)   +tau_1={d_w1_pos:+.5f}  -tau_1={d_w1_neg:+.5f}  "
        f"|ratio|={ratio_str(d_w1_pos, d_w1_neg)}  "
        f"signs={'opposite OK' if d_w1_pos * d_w1_neg < 0 else 'SAME -- BAD'}")
    log(f"Magnitudes       |Delta(dth0/dt)| under +/-tau_0: "
        f"{0.5 * (abs(d_w0_pos) + abs(d_w0_neg)):.5f}  "
        f"(should grow with epochs if WM learns torque->angular accel)")
    log(f"                 |Delta(dth1/dt)| under +/-tau_1: "
        f"{0.5 * (abs(d_w1_pos) + abs(d_w1_neg)):.5f}")

    # Cross-coupling (tau_0 affecting dth1 and vice versa).
    cross_01 = 0.5 * (abs(rel["+tau_0"][7]) + abs(rel["-tau_0"][7]))
    cross_10 = 0.5 * (abs(rel["+tau_1"][6]) + abs(rel["-tau_1"][6]))
    log(f"Cross-coupling   |Delta(dth1/dt) | +/-tau_0| = {cross_01:.5f}  "
        f"(physical: smaller than direct)")
    log(f"                 |Delta(dth0/dt) | +/-tau_1| = {cross_10:.5f}")


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline tests for the Reacher world model")
    parser.add_argument("--dataset",     default="reacher_val_dataset.npz",
                        help="Path to dataset .npz (default: val set)")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--checkpoint",  default=None,
                        help="World model checkpoint (default: world_model.pt)")
    parser.add_argument("--mode",        choices=["teacher", "open", "symmetry"], default="teacher",
                        help="teacher: posterior; open: prior rollout with GT actions; "
                             "symmetry: control-sensitivity test")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Seed for selecting episodes")
    parser.add_argument("--plot_dir",    default="plots",
                        help="If set, save plots here (default: ./plots). "
                             "Pass empty string '' to disable plotting.")
    parser.add_argument("--log_path",    default=None,
                        help="If set, tee stdout into this file for easy copy/paste.")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="Warmup steps for symmetry mode")
    parser.add_argument("--sample_every", type=int, default=5,
                        help="Sample state every N steps after warmup (symmetry mode)")
    args = parser.parse_args()

    plot_dir = args.plot_dir if args.plot_dir else None
    log = Logger(args.log_path)

    log(f"test_worldmodel (Reacher) | mode={args.mode}")
    log(f"  dataset   = {args.dataset}")
    log(f"  config    = {args.config}")
    log(f"  checkpoint= {args.checkpoint or '<auto>'}")
    log(f"  device    = {DEVICE}")
    if plot_dir:
        log(f"  plot_dir  = {plot_dir}")
    log("")

    episodes = load_episodes(args.dataset)
    if not episodes:
        log("No episodes found in dataset.")
        log.close()
        return 1
    log(f"Loaded {len(episodes)} episodes from {args.dataset}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(episodes), size=min(args.max_episodes, len(episodes)), replace=False)
    log(f"Selected episodes (seed={args.seed}): {[int(episodes[i]['ep_id']) for i in picks]}")

    latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim = load_config(args.config)
    obs_dim = episodes[0]["obs"].shape[1]
    world_model = WorldModel(
        obs_dim, action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    load_checkpoint(world_model, args.checkpoint, log)
    world_model.eval()
    log("")

    if args.mode == "symmetry":
        control_sensitivity_test(world_model, episodes, log,
                                 num_episodes=args.max_episodes,
                                 warmup_steps=args.warmup_steps,
                                 sample_every=args.sample_every,
                                 seed=args.seed)
        log.close()
        return 0

    # teacher or open: per-episode rollout + metrics + per-dim breakdown +
    # plot.  Aggregate metrics at the end across the selected episodes.
    all_obs_mae    = []
    all_obs_sincos = []
    all_reward_mae = []
    all_return_diff = []
    per_dim_mae_accum = np.zeros(obs_dim, dtype=np.float64)
    n_eps_done = 0

    for ep_idx in picks:
        ep = episodes[ep_idx]
        ep_id = ep["ep_id"]
        if args.mode == "teacher":
            obs_gt, obs_pred, rewards_gt, reward_pred = rollout_teacher(world_model, ep)
        else:
            obs_gt, obs_pred, rewards_gt, reward_pred = rollout_open(world_model, ep)

        o = compute_obs_metrics(obs_gt, obs_pred)
        r = compute_reward_metrics(rewards_gt, reward_pred)

        log(f"[Episode {ep_id}] ({args.mode}):")
        log(f"  {format_obs_metrics(o)}")
        log(f"  {format_reward_metrics(r)}")
        log(format_per_dim(o))

        if plot_dir:
            path = plot_observations(
                obs_gt, obs_pred,
                title=f"{args.mode} rollout, episode {ep_id}",
                rewards_gt=rewards_gt, reward_pred=reward_pred,
                plot_dir=plot_dir,
                filename=f"{args.mode}_ep{ep_id:04d}.png",
            )
            if path:
                log(f"  plot saved: {path}")
        log("")

        all_obs_mae.append(o["mae"])
        all_obs_sincos.append(o["sincos_mae"])
        all_reward_mae.append(r["mae"])
        all_return_diff.append(r["return_pred"] - r["return_gt"])
        per_dim_mae_accum += o["per_dim_mae"]
        n_eps_done += 1

    if n_eps_done > 0:
        log("=" * 60)
        log(f"Aggregate over {n_eps_done} episodes ({args.mode} mode):")
        log(f"  obs MAE        mean={np.mean(all_obs_mae):.4f}  "
            f"std={np.std(all_obs_mae):.4f}  max={np.max(all_obs_mae):.4f}")
        log(f"  sin/cos MAE    mean={np.mean(all_obs_sincos):.4f}  "
            f"std={np.std(all_obs_sincos):.4f}  max={np.max(all_obs_sincos):.4f}")
        log(f"  reward MAE     mean={np.mean(all_reward_mae):.4f}  "
            f"std={np.std(all_reward_mae):.4f}  max={np.max(all_reward_mae):.4f}")
        log(f"  return diff    mean={np.mean(all_return_diff):+.3f}  "
            f"std={np.std(all_return_diff):.3f}  "
            f"(pred - gt; positive = WM overestimates return)")
        log("  per-dim obs MAE (averaged over episodes):")
        for d in range(min(len(OBS_NAMES), len(per_dim_mae_accum))):
            log(f"    {OBS_NAMES[d]:<12}  {per_dim_mae_accum[d] / n_eps_done:.4f}")

    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
