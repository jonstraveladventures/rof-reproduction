"""
Diagnose the flat-failing LLC MPC (Stage-B manipulation-check failure).

Four probes on one checkpoint, separating the candidate faults:
  A. On-manifold model quality: imagined discounted return under GROUND-TRUTH
     action sequences from validation episodes vs the real discounted reward
     over the same window. Good agreement = model fine on-manifold.
  B. Planner optimism: run MPC episodes logging CEM's imagined best_score at
     each step vs the discounted return actually realised over the next H
     real steps. Large positive gap = model exploitation by the planner.
  C. Random shooting at matched budget (population*iters samples, one shot):
     if it matches CEM, search isn't the bottleneck.
  D. Heuristic pilot through the same env loop: environment-interface sanity
     (should land ~+200 clean).
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.box2d.lunar_lander import heuristic
from torch.utils.data import DataLoader

from models import actions_to_vec
from train_models import SequenceDataset
from wm_mpc_policy_continuous import DEVICE, cem_plan, evaluate_action_sequences, load_world_model


@torch.no_grad()
def probe_A(wm, args):
    cfg_action_dim = wm.rssm.action_dim
    ds = SequenceDataset(args.val_dataset, sequence_length=30, action_dim=cfg_action_dim,
                         random_start=False, dataset_seq_offset=10)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    gaps, imag, real = [], [], []
    n = 0
    for batch in loader:
        obs_seq, actions_seq, rewards_seq, next_obs_seq, _, mask = [t.to(DEVICE) for t in batch]
        B, T = obs_seq.shape[:2]
        if T < 5 + args.horizon:
            continue
        valid = mask[:, 5 + args.horizon - 1] > 0
        if not valid.any():
            continue
        h = wm.rssm.init_hidden(B, DEVICE)
        z = torch.zeros(B, wm.rssm.latent_dim, device=DEVICE)
        h = wm.rssm.update_hidden(h, z, torch.zeros(B, cfg_action_dim, device=DEVICE))
        mu, _ = wm.rssm.posterior(h, obs_seq[:, 0])
        z = mu
        for t in range(4):
            a_t = actions_to_vec(actions_seq[:, t], cfg_action_dim)
            h = wm.rssm.update_hidden(h, z, a_t)
            mu, _ = wm.rssm.posterior(h, next_obs_seq[:, t])
            z = mu
        im = torch.zeros(B, device=DEVICE)
        disc = 1.0
        hh, zz = h, z
        for k in range(args.horizon):
            a_t = actions_to_vec(actions_seq[:, 4 + k], cfg_action_dim)
            hh = wm.rssm.update_hidden(hh, zz, a_t)
            zz, _ = wm.rssm.prior(hh)
            im = im + disc * wm.predict_reward(hh, zz).squeeze(-1)
            disc *= args.gamma
        re = torch.zeros(B, device=DEVICE)
        disc = 1.0
        for k in range(args.horizon):
            re = re + disc * rewards_seq[:, 4 + k]
            disc *= args.gamma
        imag += im[valid].tolist(); real += re[valid].tolist()
        gaps += (im[valid] - re[valid]).tolist()
        n += int(valid.sum().item())
        if n >= 1500:
            break
    imag, real, gaps = np.array(imag), np.array(real), np.array(gaps)
    corr = np.corrcoef(imag, real)[0, 1]
    print(f"\n[A] on-manifold (ground-truth actions, n={len(gaps)}):")
    print(f"    imagined {imag.mean():+.2f}+-{imag.std():.2f}  real {real.mean():+.2f}+-{real.std():.2f}")
    print(f"    gap mean {gaps.mean():+.2f}  |gap| mean {np.abs(gaps).mean():.2f}  corr(imag, real) {corr:+.3f}")


class Args:
    pass


def run_mpc_episode(wm, action_dim, seed, args, mode="cem"):
    env = gym.make("LunarLander-v3", continuous=True)
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    h = wm.rssm.init_hidden(1, DEVICE)
    z = torch.zeros(1, wm.rssm.latent_dim, device=DEVICE)
    prev = np.zeros(action_dim, dtype=np.float32)
    ep_ret, steps, done = 0.0, 0, False
    imagined_scores, step_rewards = [], []
    while not done and steps < args.max_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prev_t = torch.tensor(prev, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            h, z, _, _, _, _ = wm.rssm.step(h, z, prev_t, obs_t)
            if mode == "cem":
                action, score, _ = cem_plan(wm, h, z, action_dim, args)
            else:  # random shooting, matched budget
                P = args.population * args.cem_iters
                seqs = (torch.rand(P, args.horizon, action_dim, device=DEVICE) * 2 - 1)
                scores = evaluate_action_sequences(wm, h, z, seqs, args)
                b = int(scores.argmax().item())
                action, score = seqs[b, 0].cpu().numpy(), float(scores[b].item())
        imagined_scores.append(score)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        step_rewards.append(float(reward))
        ep_ret += float(reward)
        obs = next_obs
        prev = np.asarray(action, dtype=np.float32)
        steps += 1
    env.close()
    # realised discounted return over next H steps, per step
    H, g = args.horizon, args.gamma
    realised = []
    for t in range(len(step_rewards)):
        w = step_rewards[t:t + H]
        realised.append(sum((g ** k) * r for k, r in enumerate(w)))
    im = np.array(imagined_scores); re = np.array(realised)
    return ep_ret, steps, im, re


def probe_BCD(wm, action_dim, args):
    for mode, label in (("cem", "B: CEM-MPC"), ("rand", "C: random shooting (matched budget)")):
        rets, gaps = [], []
        for ep in range(args.episodes):
            ret, steps, im, re = run_mpc_episode(wm, action_dim, args.seed + ep, args, mode)
            rets.append(ret); gaps.append((im - re).mean())
            print(f"[{label}] ep{ep+1}: return {ret:+.1f} steps {steps} "
                  f"imagined {im.mean():+.2f} realised {re.mean():+.2f} optimism-gap {(im-re).mean():+.2f}")
        print(f"[{label}] mean return {np.mean(rets):+.1f}, mean optimism gap {np.mean(gaps):+.2f}")

    env = gym.make("LunarLander-v3", continuous=True)
    rets = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ret, t = False, 0.0, 0
        while not done and t < 1000:
            a = heuristic(env.unwrapped, obs)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc; ret += float(r); t += 1
        rets.append(ret)
    env.close()
    print(f"[D: heuristic sanity] returns {[f'{r:+.0f}' for r in rets]} (env solvable check)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_llc.yaml")
    p.add_argument("--val_dataset", default="lunarlander_continuous_val_dataset.npz")
    p.add_argument("--world_model", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()

    args = Args()
    args.horizon, args.population, args.elites = 25, 384, 48
    args.cem_iters, args.cem_alpha = 4, 0.7
    args.cem_init_std, args.cem_min_std = 0.5, 0.05
    args.gamma, args.max_steps = 0.97, 600
    args.episodes, args.seed = a.episodes, a.seed
    args.val_dataset = a.val_dataset

    env = gym.make("LunarLander-v3", continuous=True)
    obs_dim = env.observation_space.shape[0]
    env.close()
    wm, action_dim = load_world_model(a.config, a.world_model, obs_dim)
    print(f"checkpoint: {a.world_model}")
    probe_A(wm, args)
    probe_BCD(wm, action_dim, args)


if __name__ == "__main__":
    main()
