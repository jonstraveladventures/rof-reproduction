"""
Scripted dataset collection for LunarLander (discrete or continuous).

Replaces human piloting with the Gymnasium built-in heuristic controller,
corrupted per-episode to produce a mix of clean landings, degraded flights,
and crashes. Calibration target is the return mix of the original
human-piloted dataset (~42% return>=+100, ~42% return<=-100, ~17% middle).

Per-episode corruption modes (drawn with --p_clean/--p_noisy/--p_blackout):
  clean:    Gaussian action noise, sigma ~ U(0, 0.10)
  noisy:    Gaussian action noise, sigma ~ U(0.40, 0.90)
  blackout: heuristic flies normally until t0 ~ U(20, 200), then the pilot
            "passes out": engines cut for the rest of the episode (free fall)
  disoriented: from t0 ~ U(0, 80), the heuristic sees x and vx sign-flipped
            and steers for the wrong side of the map (deeply negative returns)

Output npz schema matches collect_dataset.py:
  obs, next_obs (float32, [T, 8]), rewards (float32, [T]),
  dones, ep_index, episode_seed, step_index (int64, [T]),
  actions: int64 [T] (discrete) or float32 [T, 2] (continuous)
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic


def corrupt_action(a, mode_sigma, rng, blackout_active, continuous, action_space):
    if blackout_active:
        # engines off: free fall (discrete: action 0 = do nothing)
        if continuous:
            return np.array([-1.0, 0.0], dtype=np.float32)
        return 0
    if continuous:
        noisy = np.asarray(a, dtype=np.float32) + rng.normal(0.0, mode_sigma, size=2)
        return np.clip(noisy, -1.0, 1.0).astype(np.float32)
    # discrete: with probability proportional to sigma, replace with random action
    if rng.random() < min(1.0, mode_sigma):
        return action_space.sample()
    return a


def collect(args):
    env = gym.make("LunarLander-v3", continuous=args.continuous)
    rng = np.random.default_rng(args.seed)

    modes = ["clean", "noisy", "blackout", "disoriented"]
    probs = np.array([args.p_clean, args.p_noisy, args.p_blackout, args.p_disoriented], dtype=np.float64)
    probs = probs / probs.sum()

    cols = {k: [] for k in ("obs", "next_obs", "rewards", "dones",
                            "ep_index", "episode_seed", "step_index", "actions")}
    ep_returns = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        obs, _ = env.reset(seed=ep_seed)
        mode = rng.choice(modes, p=probs)
        if mode == "clean":
            sigma = rng.uniform(0.0, 0.10)
        elif mode == "noisy":
            sigma = rng.uniform(0.40, 0.90)
        else:
            sigma = rng.uniform(0.0, 0.10)
        blackout_start = int(rng.uniform(20, 200)) if mode == "blackout" else -1
        disorient_start = int(rng.uniform(0, 80)) if mode == "disoriented" else -1

        ep_ret, t, done = 0.0, 0, False
        while not done and t < args.max_steps:
            obs_for_pilot = obs
            if disorient_start >= 0 and t >= disorient_start:
                obs_for_pilot = obs.copy()
                obs_for_pilot[0] = -obs_for_pilot[0]
                obs_for_pilot[2] = -obs_for_pilot[2]
            a = heuristic(env.unwrapped, obs_for_pilot)
            blackout_active = blackout_start >= 0 and t >= blackout_start
            a = corrupt_action(a, sigma, rng, blackout_active, args.continuous, env.action_space)

            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            cols["obs"].append(obs)
            cols["next_obs"].append(next_obs)
            cols["rewards"].append(reward)
            cols["dones"].append(int(done))
            cols["ep_index"].append(ep)
            cols["episode_seed"].append(ep_seed)
            cols["step_index"].append(t)
            cols["actions"].append(a)

            obs = next_obs
            ep_ret += reward
            t += 1
        ep_returns.append(ep_ret)

    env.close()
    rets = np.array(ep_returns)
    good, bad = (rets >= 100).mean(), (rets <= -100).mean()
    print(f"episodes={len(rets)} steps={len(cols['rewards'])} "
          f"mean_len={len(cols['rewards'])/len(rets):.0f}")
    print(f"return: mean {rets.mean():+.1f} median {np.median(rets):+.1f} "
          f"min {rets.min():+.1f} max {rets.max():+.1f}")
    print(f"mix: good(>=100) {good:.1%}  bad(<=-100) {bad:.1%}  middle {1-good-bad:.1%}")

    if args.dataset:
        out = {}
        for k, v in cols.items():
            arr = np.asarray(v)
            if k in ("obs", "next_obs", "rewards"):
                arr = arr.astype(np.float32)
            elif k == "actions":
                arr = arr.astype(np.float32) if args.continuous else arr.astype(np.int64)
            else:
                arr = arr.astype(np.int64)
            out[k] = arr
        np.savez_compressed(args.dataset, **out)
        print(f"saved -> {args.dataset}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="output .npz (omit for pilot/mix check)")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--continuous", type=lambda s: s.lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--p_clean", type=float, default=0.42)
    p.add_argument("--p_noisy", type=float, default=0.15)
    p.add_argument("--p_blackout", type=float, default=0.10)
    p.add_argument("--p_disoriented", type=float, default=0.33)
    collect(p.parse_args())
