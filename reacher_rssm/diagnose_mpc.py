# ===========================================================================
# Reacher RSSM - diagnostic script for MPC failure investigation.
#
# Five tests, run against one world-model checkpoint:
#
#   [1] 1-step prediction accuracy.
#       Roll one real episode, posterior-step the WM at every env step, and
#       compare (a) predicted reward to actual reward, (b) predicted next-obs
#       to actual next-obs. Low error here = WM training is roughly OK.
#
#   [2] Action-sensitivity probe at the start state.
#       At t=0 (h, z) after observing obs_reset, score 8 fixed action
#       sequences over the planning horizon (zero, +1, -1, joint-0 only,
#       joint-1 only, alternating signs). If the spread of predicted returns
#       across these very different policies is tiny, the WM has no useful
#       action gradient and MPC cannot work no matter how it is tuned.
#
#   [3] Prior-rollout drift.
#       Same start state, but for each canonical sequence ALSO run the
#       sequence in the real env. Compare predicted vs actual return.
#       Reveals whether the prior diverges from reality over the horizon.
#
#   [4] Zero-action baseline.
#       10 episodes, every action = (0, 0). Pure floor.
#
#   [5] IK-oracle baseline.
#       10 episodes using the same IK + PD controller used for "good_ik" data
#       collection. This is the oracle upper bound; MPC should approach it
#       if both the WM and the planner are healthy.
#
# Usage:
#   python diagnose_mpc.py --world_model checkpoints/world_model_..._epoch_445.pt
# ===========================================================================

from __future__ import annotations
import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import yaml

from models import WorldModel
from ik_controller import (
    inverse_kinematics_2link,
    pd_torque,
    joint_angles_from_obs,
    joint_velocities_from_obs,
    target_from_obs,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "Reacher-v5"
EPISODE_LENGTH = 100
LINK_LENGTH_1 = 0.10
LINK_LENGTH_2 = 0.11
PD_KP = 2.0
PD_KD = 0.3


# ---------------------------------------------------------------------------
# Setup helpers (copied from wm_mpc_policy.py to keep this file self-contained).
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    if not os.path.exists(config_path):
        return 16, 256, 256, 1, 2
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm_cfg = config.get("world_model", {})
    cap = wm_cfg.get("capacity", {})
    return (
        int(cap.get("latent_dim",     16)),
        int(cap.get("hidden_dim",     256)),
        int(cap.get("mlp_hidden_dim", cap.get("hidden_dim", 256))),
        int(cap.get("gru_num_layers", 1)),
        int(wm_cfg.get("action_dim",  2)),
    )


def load_world_model(config_path: str, checkpoint_path: str, obs_dim: int):
    latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim = load_config(config_path)
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    world_model.load_state_dict(state_dict)
    world_model.eval()
    return world_model, action_dim


def ik_pd_action(obs: np.ndarray) -> np.ndarray:
    target = target_from_obs(obs)
    theta0_t, theta1_t = inverse_kinematics_2link(
        float(target[0]), float(target[1]),
        l1=LINK_LENGTH_1, l2=LINK_LENGTH_2, elbow="up",
    )
    theta_now = joint_angles_from_obs(obs)
    theta_dot = joint_velocities_from_obs(obs)
    return pd_torque(
        theta_current=theta_now, theta_dot=theta_dot,
        theta_target=np.array([theta0_t, theta1_t], dtype=np.float64),
        kp=PD_KP, kd=PD_KD,
        action_low=-1.0, action_high=1.0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# [1] One-step prediction accuracy.
# ---------------------------------------------------------------------------
@torch.no_grad()
def diag_one_step_accuracy(world_model, args):
    print("\n" + "=" * 72)
    print("[1] 1-step prediction accuracy (IK policy, 1 episode)")
    print("=" * 72)
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    set_seed(args.seed)

    obs_dim = world_model.rssm.obs_dim
    action_dim = world_model.rssm.action_dim
    h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
    z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
    prev_action = np.zeros(action_dim, dtype=np.float32)

    abs_err_reward = []
    abs_err_obs = []
    abs_err_target_offset = []   # obs[8:10] -- the part that dominates reward
    # Reacher obs layout has (cos, sin) at non-adjacent indices:
    #   shoulder: (obs[0], obs[2])
    #   elbow:    (obs[1], obs[3])
    # The WM is trained with plain MSE which does NOT constrain these to the
    # unit circle; over a long prior rollout the predicted (cos, sin) can
    # drift off the circle. Track per-step deviation here.
    unit_circle_dev = []

    for step in range(1, args.max_steps + 1):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_a_t = torch.tensor(prev_action, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Posterior-step from current obs (this matches training bootstrap on step 1
        # and training iter t>=1 on subsequent steps).
        h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_a_t, obs_t)

        action = ik_pd_action(obs)

        # Predict the *next* (h', z') under prior + chosen action; then read off
        # predicted reward and next obs.
        a_t = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        h_next = world_model.rssm.update_hidden(h, z, a_t)
        mean_prior, _ = world_model.rssm.prior(h_next)
        z_next_prior = mean_prior
        r_hat = float(world_model.predict_reward(h_next, z_next_prior).item())
        o_hat = world_model.decode_obs(h_next, z_next_prior).squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)

        abs_err_reward.append(abs(r_hat - float(reward)))
        abs_err_obs.append(np.mean(np.abs(o_hat - next_obs)))
        abs_err_target_offset.append(np.mean(np.abs(o_hat[8:10] - next_obs[8:10])))

        norm_shoulder = float(np.sqrt(o_hat[0] ** 2 + o_hat[2] ** 2))
        norm_elbow    = float(np.sqrt(o_hat[1] ** 2 + o_hat[3] ** 2))
        unit_circle_dev.append(max(abs(norm_shoulder - 1.0), abs(norm_elbow - 1.0)))

        obs = next_obs
        prev_action = action
        if terminated or truncated:
            break

    env.close()
    print(f"  reward         MAE = {np.mean(abs_err_reward):.4f}  "
          f"(typical reward magnitude ~ 0.05..0.5)")
    print(f"  obs (all)      MAE = {np.mean(abs_err_obs):.4f}  per-dim avg over {obs_dim} dims")
    print(f"  obs[8:10]      MAE = {np.mean(abs_err_target_offset):.4f}  "
          f"(fingertip-target offset; drives reward)")
    print(f"  sin/cos drift  mean={np.mean(unit_circle_dev):.4f}  max={np.max(unit_circle_dev):.4f}  "
          f"(|norm-1| for (cos,sin) pairs; should be << 0.05)")
    if np.mean(abs_err_target_offset) < 0.02:
        print("  Verdict: WM 1-step obs prediction is TIGHT for the reward-driving dims.")
    elif np.mean(abs_err_target_offset) < 0.05:
        print("  Verdict: WM 1-step obs prediction is OK but not great.")
    else:
        print("  Verdict: WM 1-step obs prediction is POOR -- WM is the bottleneck.")
    if np.max(unit_circle_dev) > 0.1:
        print("  Note: sin/cos pairs drift off the unit circle even at 1-step horizon.")


# ---------------------------------------------------------------------------
# [2] Action sensitivity at the start state.
# ---------------------------------------------------------------------------
def canonical_action_sequences(horizon: int, action_dim: int):
    """Return list of (name, sequence-tensor) pairs of shape (horizon, action_dim)."""
    H, A = horizon, action_dim
    z   = torch.zeros(H, A, device=DEVICE)
    pos = torch.ones(H, A, device=DEVICE)
    neg = -torch.ones(H, A, device=DEVICE)
    j0p = torch.zeros(H, A, device=DEVICE); j0p[:, 0] = 1.0
    j0n = torch.zeros(H, A, device=DEVICE); j0n[:, 0] = -1.0
    j1p = torch.zeros(H, A, device=DEVICE); j1p[:, 1] = 1.0
    j1n = torch.zeros(H, A, device=DEVICE); j1n[:, 1] = -1.0
    alt = torch.ones(H, A, device=DEVICE)
    for t in range(H):
        if t % 2 == 1:
            alt[t] *= -1.0
    return [
        ("zero",       z),
        ("all +1",     pos),
        ("all -1",     neg),
        ("j0 +1, j1 0", j0p),
        ("j0 -1, j1 0", j0n),
        ("j0 0, j1 +1", j1p),
        ("j0 0, j1 -1", j1n),
        ("alt +/- 1",  alt),
    ]


@torch.no_grad()
def score_sequence(world_model, h0, z0, seq, gamma):
    """Roll one (H, A) sequence through the prior; return discounted summed predicted reward."""
    h = h0.clone(); z = z0.clone()
    score = 0.0
    discount = 1.0
    for t in range(seq.shape[0]):
        a_t = seq[t].unsqueeze(0)
        h = world_model.rssm.update_hidden(h, z, a_t)
        mean_prior, _ = world_model.rssm.prior(h)
        z = mean_prior
        r = float(world_model.predict_reward(h, z).item())
        score += discount * r
        discount *= gamma
    return score


@torch.no_grad()
def rollout_sequence_with_decode(world_model, h0, z0, seq, gamma):
    """Like score_sequence but also returns the worst sin/cos unit-circle
    deviation over the horizon, and the discounted score."""
    h = h0.clone(); z = z0.clone()
    score = 0.0
    discount = 1.0
    worst_dev = 0.0
    for t in range(seq.shape[0]):
        a_t = seq[t].unsqueeze(0)
        h = world_model.rssm.update_hidden(h, z, a_t)
        mean_prior, _ = world_model.rssm.prior(h)
        z = mean_prior
        r = float(world_model.predict_reward(h, z).item())
        score += discount * r
        discount *= gamma

        o_pred = world_model.decode_obs(h, z).squeeze(0).cpu().numpy()
        norm_shoulder = float(np.sqrt(o_pred[0] ** 2 + o_pred[2] ** 2))
        norm_elbow    = float(np.sqrt(o_pred[1] ** 2 + o_pred[3] ** 2))
        worst_dev = max(worst_dev, abs(norm_shoulder - 1.0), abs(norm_elbow - 1.0))
    return score, worst_dev


@torch.no_grad()
def diag_action_sensitivity(world_model, args):
    print("\n" + "=" * 72)
    print(f"[2] Action sensitivity at start state (horizon={args.horizon}, gamma={args.gamma})")
    print("=" * 72)
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    action_dim = world_model.rssm.action_dim

    h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
    z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
    prev_a_t = torch.zeros(1, action_dim, device=DEVICE)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    h0, z0, _, _, _, _ = world_model.rssm.step(h, z, prev_a_t, obs_t)
    env.close()

    sequences = canonical_action_sequences(args.horizon, action_dim)
    print(f"  {'sequence':<14}  predicted_return")
    scores = []
    for name, seq in sequences:
        s = score_sequence(world_model, h0, z0, seq, args.gamma)
        scores.append(s)
        print(f"  {name:<14}  {s:+8.3f}")

    scores = np.array(scores)
    spread = scores.max() - scores.min()
    print(f"  spread = max - min = {spread:.3f}")
    if spread < 0.5:
        print("  Verdict: WM is action-INSENSITIVE. MPC cannot work with this model.")
    elif spread < 2.0:
        print("  Verdict: WM has weak action sensitivity. MPC will be noisy.")
    else:
        print("  Verdict: WM has meaningful action sensitivity.")


# ---------------------------------------------------------------------------
# [3] Prior-rollout drift vs reality.
# ---------------------------------------------------------------------------
@torch.no_grad()
def diag_prior_drift(world_model, args):
    print("\n" + "=" * 72)
    print(f"[3] Prior-rollout drift over {args.horizon} steps")
    print("=" * 72)
    action_dim = world_model.rssm.action_dim
    sequences = canonical_action_sequences(args.horizon, action_dim)

    print(f"  {'sequence':<14}  pred_return   real_return    diff    sin/cos max-dev")
    for name, seq in sequences:
        # Establish (h0, z0) from a fresh reset.
        env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=args.max_steps)
        obs, _ = env.reset(seed=args.seed)

        h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
        z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        prev_a_t = torch.zeros(1, action_dim, device=DEVICE)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        h0, z0, _, _, _, _ = world_model.rssm.step(h, z, prev_a_t, obs_t)

        pred_return, worst_dev = rollout_sequence_with_decode(
            world_model, h0, z0, seq, args.gamma
        )

        # Roll the same sequence in the real env (discounted to compare apples
        # to discounted -- we want directional agreement, not absolute match).
        real_return_disc = 0.0
        discount = 1.0
        for t in range(args.horizon):
            action = seq[t].cpu().numpy()
            _, reward, term, trunc, _ = env.step(action)
            real_return_disc += discount * float(reward)
            discount *= args.gamma
            if term or trunc:
                break
        env.close()

        diff = pred_return - real_return_disc
        print(f"  {name:<14}  {pred_return:+8.3f}     {real_return_disc:+8.3f}     "
              f"{diff:+7.3f}    {worst_dev:.3f}")

    print("  (real_return is gamma-discounted to match the predicted_return convention)")
    print("  (sin/cos max-dev is the worst |norm-1| seen anywhere in the H-step rollout)")


# ---------------------------------------------------------------------------
# [4][5] Open-loop baselines (no MPC, no WM).
# ---------------------------------------------------------------------------
def run_baseline(policy_fn, label: str, n_episodes: int, seed: int, max_steps: int):
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=max_steps)
    returns = []
    final_dists = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep + 1)
        ep_return = 0.0
        for _ in range(max_steps):
            action = policy_fn(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            obs = next_obs
            if term or trunc:
                break
        returns.append(ep_return)
        final_dists.append(float(np.sqrt(obs[8] ** 2 + obs[9] ** 2)))
    env.close()
    rs = np.array(returns)
    ds = np.array(final_dists)
    print(f"  {label:<14}  return mean={rs.mean():+7.2f}  std={rs.std():5.2f}  "
          f"min={rs.min():+7.2f}  max={rs.max():+7.2f}  "
          f"final_dist mean={ds.mean():.4f}")
    return rs, ds


def diag_baselines(args):
    print("\n" + "=" * 72)
    print(f"[4][5] Open-loop baselines (n={args.baseline_episodes} episodes each)")
    print("=" * 72)
    rng = np.random.default_rng(args.seed ^ 0xC0FFEE)

    def zero_policy(_obs):
        return np.zeros(2, dtype=np.float32)

    def random_policy(_obs):
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    def ik_policy(obs):
        return ik_pd_action(obs)

    print(f"  {'policy':<14}  result")
    run_baseline(zero_policy,   "zero action",   args.baseline_episodes, args.seed, args.max_steps)
    run_baseline(random_policy, "random U[-1,1]", args.baseline_episodes, args.seed, args.max_steps)
    run_baseline(ik_policy,     "IK + PD",       args.baseline_episodes, args.seed, args.max_steps)
    print("  (IK + PD is the oracle controller used for good_ik data; "
          "MPC should approach this if WM + planner are healthy.)")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MPC failure diagnostic for the Reacher world model")
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--world_model",  required=True)
    parser.add_argument("--seed",         type=int,   default=12345)
    parser.add_argument("--max_steps",    type=int,   default=EPISODE_LENGTH)
    parser.add_argument("--horizon",      type=int,   default=25)
    parser.add_argument("--gamma",        type=float, default=0.97)
    parser.add_argument("--baseline_episodes", type=int, default=10)
    parser.add_argument("--skip",         nargs="*", default=[],
                        choices=["1step", "sensitivity", "drift", "baselines"],
                        help="Skip selected sections (useful for fast iteration)")
    args = parser.parse_args()

    # Need an env briefly to read obs_dim.
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    env.close()
    world_model, _ = load_world_model(args.config, args.world_model, obs_dim)
    print(f"Loaded WM: {args.world_model}")
    print(f"Device: {DEVICE}")

    if "1step"       not in args.skip: diag_one_step_accuracy(world_model, args)
    if "sensitivity" not in args.skip: diag_action_sensitivity(world_model, args)
    if "drift"       not in args.skip: diag_prior_drift(world_model, args)
    if "baselines"   not in args.skip: diag_baselines(args)


if __name__ == "__main__":
    main()
