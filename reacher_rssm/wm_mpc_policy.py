# ===========================================================================
# Reacher RSSM - World-model MPC (Gaussian CEM) for Reacher-v5.
#
# Differences from the lander wm_mpc_policy.py:
#   - Gaussian CEM over continuous 2-D actions (lander used discrete
#     Categorical CEM over 4 actions).
#   - Drops the hand-crafted observation_cost (lander's landing-specific
#     cost; lander defaulted --obs_cost OFF anyway, so this matches their
#     actual sweep). MPC uses only the learned reward head.
#   - Drops the done penalty / done head (Reacher has no terminations).
#   - Drops the pygame display loop; uses MuJoCo's built-in render_mode
#     ("human") when --render is passed.
#
# This file evaluates ONE world-model checkpoint. The batch sweep over
# many checkpoints lives in eval_mpc.py (ported next).
#
# CLI:
#   python wm_mpc_policy.py --world_model checkpoints/world_model_..._epoch_445.pt
#                           --episodes 20 --seed 12345
# ===========================================================================

from __future__ import annotations
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from models import WorldModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "Reacher-v5"
EPISODE_LENGTH = 100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    """Return (latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim)."""
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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"World model checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    world_model.load_state_dict(state_dict)
    world_model.eval()
    return world_model, action_dim


def expand_hidden(hidden: torch.Tensor, batch_size: int) -> torch.Tensor:
    if hidden.dim() == 2:
        return hidden.expand(batch_size, -1).contiguous()
    return hidden.expand(batch_size, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Gaussian CEM
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_action_sequences(world_model, h0, z0, action_sequences, args):
    """Roll candidate action sequences through the world model and score by
    discounted predicted reward. Action sequences are (N, H, action_dim) float."""
    batch_size = action_sequences.shape[0]
    horizon    = action_sequences.shape[1]
    h = expand_hidden(h0, batch_size)
    z = z0.expand(batch_size, -1).contiguous()

    scores = torch.zeros(batch_size, device=DEVICE)
    discount = 1.0
    for t in range(horizon):
        actions_t = action_sequences[:, t]                  # (N, action_dim) continuous
        h = world_model.rssm.update_hidden(h, z, actions_t)
        mean_prior, _ = world_model.rssm.prior(h)
        z = mean_prior
        reward_pred = world_model.predict_reward(h, z)
        scores = scores + discount * reward_pred
        discount *= args.gamma
    return scores


@torch.no_grad()
def cem_plan(world_model, h0, z0, action_dim: int, args):
    """One CEM planning call. Returns (selected_action, best_score, best_step0_reward).
    selected_action is a numpy float32 array of shape (action_dim,)."""
    mu    = torch.zeros(args.horizon, action_dim, device=DEVICE)
    sigma = torch.full((args.horizon, action_dim), args.sigma_init, device=DEVICE)

    best_score = None
    best_sequence = None

    for _ in range(args.cem_iters):
        # Sample population: shape (N, H, A)
        eps = torch.randn(args.population, args.horizon, action_dim, device=DEVICE)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        samples = samples.clamp(args.action_low, args.action_high)

        scores = evaluate_action_sequences(world_model, h0, z0, samples, args)

        top_score, top_idx = torch.max(scores, dim=0)
        if best_score is None or top_score.item() > best_score:
            best_score = float(top_score.item())
            best_sequence = samples[top_idx].clone()

        elite_k = min(args.elites, args.population)
        elite_idx = torch.topk(scores, k=elite_k, dim=0).indices
        elites = samples[elite_idx]                          # (E, H, A)

        elite_mean = elites.mean(dim=0)                      # (H, A)
        elite_std  = elites.std(dim=0)                       # (H, A)

        mu    = args.cem_alpha * elite_mean + (1.0 - args.cem_alpha) * mu
        sigma = args.cem_alpha * elite_std  + (1.0 - args.cem_alpha) * sigma
        sigma = sigma.clamp_min(args.sigma_min)

    if best_sequence is None:
        best_sequence = torch.zeros(args.horizon, action_dim, device=DEVICE)
        best_score = float("-inf")

    selected_action = best_sequence[0].detach().cpu().numpy().astype(np.float32)

    # First-step diagnostic (just the predicted reward at the chosen action).
    a0 = best_sequence[0].unsqueeze(0)                       # (1, A)
    h1 = world_model.rssm.update_hidden(h0, z0, a0)
    z1, _ = world_model.rssm.prior(h1)
    reward1 = world_model.predict_reward(h1, z1)
    best_step0_reward = float(reward1.item())

    return selected_action, best_score, best_step0_reward


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MPC policy using world model + Gaussian CEM on Reacher-v5"
    )
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--world_model",  default="world_model.pt",
                        help="World model checkpoint path")
    parser.add_argument("--episodes",     type=int, default=5)
    parser.add_argument("--seed",         type=int, default=12345)
    parser.add_argument("--max_steps",    type=int, default=EPISODE_LENGTH,
                        help=f"Max steps per episode (default {EPISODE_LENGTH})")

    # Render options
    parser.add_argument("--render", action="store_true",
                        help="Open a MuJoCo viewer window during evaluation")
    parser.add_argument("--viewer_width",  type=int, default=1024)
    parser.add_argument("--viewer_height", type=int, default=1024)

    # CEM hyperparameters (match DESIGN.md section 6)
    parser.add_argument("--horizon",     type=int,   default=25)
    parser.add_argument("--population",  type=int,   default=384)
    parser.add_argument("--elites",      type=int,   default=48)
    parser.add_argument("--cem_iters",   type=int,   default=4)
    parser.add_argument("--cem_alpha",   type=float, default=0.7,
                        help="Smoothing between elite stats and previous (mu, sigma)")
    parser.add_argument("--gamma",       type=float, default=0.97)
    parser.add_argument("--sigma_init",  type=float, default=0.5)
    parser.add_argument("--sigma_min",   type=float, default=0.05)
    parser.add_argument("--action_low",  type=float, default=-1.0)
    parser.add_argument("--action_high", type=float, default=1.0)
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if args.population < 2:
        raise ValueError("population must be >= 2")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("elites must be in [1, population]")
    if args.sigma_min <= 0.0:
        raise ValueError("sigma_min must be > 0")

    # ---- Env + model ----
    if args.render:
        env = gym.make(ENV_NAME, render_mode="human",
                       max_episode_steps=args.max_steps,
                       width=args.viewer_width, height=args.viewer_height)
    else:
        env = gym.make(ENV_NAME, render_mode=None,
                       max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    world_model, action_dim = load_world_model(args.config, args.world_model, obs_dim)

    # Seed AFTER all init so CEM sampling is deterministic regardless of how
    # many random draws gym / model-init consumed.
    set_seed(args.seed)

    print(
        f"MPC (Gaussian CEM) on {ENV_NAME}\n"
        f"  world_model={args.world_model}\n"
        f"  episodes={args.episodes}, max_steps={args.max_steps}, seed={args.seed}, device={DEVICE}\n"
        f"  CEM: H={args.horizon}, N={args.population}, E={args.elites}, "
        f"iters={args.cem_iters}, alpha={args.cem_alpha}, gamma={args.gamma}\n"
        f"  Gaussian: sigma_init={args.sigma_init}, sigma_min={args.sigma_min}, "
        f"action_bounds=[{args.action_low}, {args.action_high}]"
    )

    run_returns: list[float] = []
    run_final_distances: list[float] = []
    run_saturation: list[float] = []
    total_planner_time = 0.0
    total_planner_steps = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        set_seed(args.seed + ep)
        ep_return = 0.0
        n_saturated = 0
        n_steps = 0
        planner_score_sum = 0.0
        planner_step0_reward_sum = 0.0

        h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
        z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        prev_action = np.zeros(action_dim, dtype=np.float32)
        t0 = time.perf_counter()

        for step in range(1, args.max_steps + 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_action_t = torch.tensor(prev_action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_action_t, obs_t)
                plan_t0 = time.perf_counter()
                action, best_score, best_step0_reward = cem_plan(
                    world_model, h, z, action_dim, args
                )
                total_planner_time += time.perf_counter() - plan_t0
                total_planner_steps += 1

            planner_score_sum += float(best_score)
            planner_step0_reward_sum += float(best_step0_reward)
            n_saturated += int(np.any(np.abs(action) > 0.99))
            n_steps += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            obs = next_obs
            prev_action = action
            if terminated or truncated:
                break

        # End-of-episode diagnostics.
        # In Reacher-v5 obs, indices 8 and 9 are (fingertip - target) in x and y.
        final_distance = float(np.sqrt(obs[8] ** 2 + obs[9] ** 2)) if obs.shape[0] >= 10 else float("nan")
        saturation_pct = 100.0 * n_saturated / max(n_steps, 1)
        elapsed = time.perf_counter() - t0

        print(
            f"[Episode {ep:>3}] return={ep_return:+8.2f}  "
            f"final_dist={final_distance:.4f}  "
            f"saturated={saturation_pct:5.1f}%  "
            f"planner_avg_score={planner_score_sum / max(n_steps, 1):+7.3f}  "
            f"step0_reward_avg={planner_step0_reward_sum / max(n_steps, 1):+7.3f}  "
            f"wall={elapsed:5.1f}s"
        )

        run_returns.append(ep_return)
        run_final_distances.append(final_distance)
        run_saturation.append(saturation_pct)

    env.close()

    if run_returns:
        rs = np.array(run_returns, dtype=np.float64)
        ds = np.array(run_final_distances, dtype=np.float64)
        ss = np.array(run_saturation, dtype=np.float64)
        per_step_planner_ms = 1000.0 * total_planner_time / max(total_planner_steps, 1)
        print(
            "\n[Run] ===== Summary =====\n"
            f"[Run] return       mean={rs.mean():+7.2f}  std={rs.std():5.2f}  "
            f"min={rs.min():+7.2f}  max={rs.max():+7.2f}\n"
            f"[Run] final_dist   mean={ds.mean():7.4f}  std={ds.std():6.4f}  "
            f"min={ds.min():7.4f}  max={ds.max():7.4f}\n"
            f"[Run] saturation   mean={ss.mean():5.1f}%  (per-step actions hitting +/-1)\n"
            f"[Run] planner cost {per_step_planner_ms:.1f} ms/step "
            f"({total_planner_steps} planning steps)\n"
            f"[Run] checkpoint   {os.path.basename(args.world_model)}"
        )


if __name__ == "__main__":
    main()
