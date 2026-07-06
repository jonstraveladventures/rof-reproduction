"""
Headless CEM-MPC evaluation for continuous-action LunarLander world models.

Gaussian CEM analogue of wm_mpc_policy.py's categorical planner, at matched
planning budget (horizon 25, population 384, elites 48, iterations 4,
smoothing 0.7, discount 0.97). Belief tracking mirrors the discrete driver:
each real step the RSSM posterior is updated with (prev_action, obs), then
CEM plans open-loop through the prior. The discrete driver's landed shortcut
(idle once legs are down and still) maps to engines-off [0, 0].

Two modes:
  single:  --world_model path.pt
  sweep:   --checkpoints_dir DIR [--epoch_min E] [--epoch_max E]
Log format matches logs/mpc_eval_logs.txt ("[i/N] Epoch E -- file",
"[Episode k] return=..., steps=...") so existing parsers work.
"""

import argparse
import glob
import os
import re
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from models import WorldModel

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


def load_config(config_path):
    cfg = yaml.safe_load(open(config_path))["world_model"]
    cap = cfg["capacity"]
    return (cap["latent_dim"], cap["hidden_dim"], cap["mlp_hidden_dim"],
            cfg.get("capacity", {}).get("gru_num_layers", 1), int(cfg["action_dim"]))


def load_world_model(config_path, checkpoint_path, obs_dim):
    latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim = load_config(config_path)
    wm = WorldModel(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim,
                    hidden_dim=hidden_dim, gru_num_layers=gru_num_layers,
                    mlp_hidden_dim=mlp_hidden_dim).to(DEVICE)
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    wm.load_state_dict(sd)
    wm.eval()
    return wm, action_dim


@torch.no_grad()
def evaluate_action_sequences(world_model, h0, z0, action_sequences, args):
    """action_sequences: [P, H, A] in [-1, 1]. Returns discounted scores [P]."""
    P = action_sequences.size(0)
    h = h0.expand(P, *h0.shape[1:]).contiguous() if h0.dim() == 2 else \
        h0.expand(P, *h0.shape[1:]).contiguous()
    z = z0.expand(P, -1).contiguous()
    scores = torch.zeros(P, device=DEVICE)
    discount = 1.0
    for k in range(args.horizon):
        a_k = action_sequences[:, k]
        h = world_model.rssm.update_hidden(h, z, a_k)
        z, _ = world_model.rssm.prior(h)
        reward = world_model.predict_reward(h, z).squeeze(-1)
        discount *= args.gamma
        scores += discount * reward
    return scores


@torch.no_grad()
def cem_plan(world_model, h0, z0, action_dim, args):
    mu = torch.zeros(args.horizon, action_dim, device=DEVICE)
    sigma = torch.full((args.horizon, action_dim), args.cem_init_std, device=DEVICE)
    best_sequence, best_score = None, float("-inf")
    for _ in range(args.cem_iters):
        noise = torch.randn(args.population, args.horizon, action_dim, device=DEVICE)
        seqs = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(-1.0, 1.0)
        scores = evaluate_action_sequences(world_model, h0, z0, seqs, args)
        elite_idx = scores.topk(args.elites).indices
        elite_seqs = seqs[elite_idx]
        top_score, top_i = scores[elite_idx[0]].item(), elite_idx[0]
        if top_score > best_score:
            best_score, best_sequence = top_score, seqs[top_i].clone()
        mu = args.cem_alpha * elite_seqs.mean(dim=0) + (1.0 - args.cem_alpha) * mu
        sigma = (args.cem_alpha * elite_seqs.std(dim=0, unbiased=False)
                 + (1.0 - args.cem_alpha) * sigma).clamp_min(args.cem_min_std)
    return best_sequence[0].cpu().numpy(), best_score


def run_episodes(world_model, action_dim, args):
    env = gym.make("LunarLander-v3", continuous=True)
    returns = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        torch.manual_seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
        z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        prev_action = np.zeros(action_dim, dtype=np.float32)
        ep_return, steps, done = 0.0, 0, False
        t0 = time.perf_counter()
        while not done and steps < args.max_steps:
            landed = (len(obs) >= 8 and obs[6] >= 0.5 and obs[7] >= 0.5
                      and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_a = torch.tensor(prev_action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_a, obs_t)
            if landed:
                action = np.zeros(action_dim, dtype=np.float32)
            else:
                action, _ = cem_plan(world_model, h, z, action_dim, args)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            obs = next_obs
            prev_action = np.asarray(action, dtype=np.float32)
            steps += 1
        returns.append(ep_return)
        print(f"[Episode {ep}] return={ep_return:.2f}, steps={steps}, done={done}, "
              f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    env.close()
    rets = np.array(returns)
    print(f"[Checkpoint summary] mean={rets.mean():.2f} median={np.median(rets):.2f} "
          f"min={rets.min():.2f} max={rets.max():.2f} n={len(rets)}", flush=True)
    return rets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_llc.yaml")
    p.add_argument("--world_model", default=None, help="single checkpoint path")
    p.add_argument("--checkpoints_dir", default=None, help="sweep mode directory")
    p.add_argument("--epoch_min", type=int, default=None)
    p.add_argument("--epoch_max", type=int, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--population", type=int, default=384)
    p.add_argument("--elites", type=int, default=48)
    p.add_argument("--cem_iters", type=int, default=4)
    p.add_argument("--cem_alpha", type=float, default=0.7)
    p.add_argument("--cem_init_std", type=float, default=0.5)
    p.add_argument("--cem_min_std", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.97)
    args = p.parse_args()

    env = gym.make("LunarLander-v3", continuous=True)
    obs_dim = env.observation_space.shape[0]
    env.close()

    if args.checkpoints_dir:
        paths = []
        for f in glob.glob(os.path.join(args.checkpoints_dir, "*.pt")):
            m = re.search(r"epoch_(\d+)\.pt$", f)
            if not m:
                continue
            e = int(m.group(1))
            if args.epoch_min is not None and e < args.epoch_min:
                continue
            if args.epoch_max is not None and e > args.epoch_max:
                continue
            paths.append((e, f))
        paths.sort()
        print(f"\nMPC CEM Evaluation Sweep (continuous)\n"
              f"Checkpoints: {len(paths)}, Episodes per checkpoint: {args.episodes}, "
              f"Seed: {args.seed}\nDevice: {DEVICE}\n", flush=True)
        for i, (e, f) in enumerate(paths, 1):
            print("=" * 70, flush=True)
            print(f"  [{i}/{len(paths)}] Epoch {e} -- {os.path.basename(f)}", flush=True)
            print("=" * 70, flush=True)
            wm, action_dim = load_world_model(args.config, f, obs_dim)
            run_episodes(wm, action_dim, args)
    else:
        wm, action_dim = load_world_model(args.config, args.world_model, obs_dim)
        run_episodes(wm, action_dim, args)


if __name__ == "__main__":
    main()
