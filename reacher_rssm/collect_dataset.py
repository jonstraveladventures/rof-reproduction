# ===========================================================================
# Reacher RSSM - dataset collection.
#
# Single-file pipeline producing reacher_train_dataset.npz and
# reacher_val_dataset.npz (dataset_version=2 in the NPZ metadata) with six
# buckets designed to cover the full reward range with REALISTIC movement:
#
#   class 0  smooth_ik         (35%)  IK + PD, kp=2, kd=0.3, sigma=0.05
#                                     -> the current "good_ik" behavior.
#   class 1  aggressive_ik     (20%)  IK + P-only, kp=8, kd=0 (both configurable).
#                                     Pure P at high gain saturates ~80% of steps
#                                     and yields ACF(1) ~ +0.85 (smooth sustained
#                                     boundary signal). Adding D destabilises the
#                                     saturated regime: kd=0.5 gives ACF(1)~-0.6
#                                     (chattering), kd>=1 gives ACF(1)<-0.9.
#                                     See calibration sweep results.
#   class 2  wrong_target_ik   (15%)  IK + PD aimed at a per-episode random
#                                     point in the radius-0.21 disk (NOT the
#                                     env's target). Smooth motion to a wrong
#                                     destination. Fills the mid-return gap.
#   class 3  block_random      (15%)  Per-episode amplitude A in {0.3, 0.7, 1.0};
#                                     per-block (K=10 steps) action ~ U(-A, A)
#                                     held constant. 10 blocks per episode.
#                                     Matches CEM's "sustained narrow Gaussian"
#                                     behavior.
#   class 4  iid_random        (10%)  U(-1, 1) iid per step. Pure noise.
#   class 5  zero              ( 5%)  (0, 0) constant. "No torque -> coast."
#
# Velocity safety: aggressive_ik and block_random pass actions through a
# velocity-clipping filter that opposes joint motion when |theta_dot| exceeds
# --velocity_limit rad/s (default 80). Prevents the shoulder from spinning
# beyond ~80 rad/s during sustained boundary actions.
#
# Episode length is uniform 100 steps for all buckets.
#
# CLI:
#   # 1. (Optional) Calibrate the aggressive_ik kp value:
#   python collect_dataset.py --calibrate_kp
#
#   # 2. Full collection (~45 s headless, ~40 ep/s):
#   python collect_dataset.py --headless --train_episodes 1500 --val_episodes 240
#
# On Ctrl+C the script saves whatever has been collected so far.
# ===========================================================================

from __future__ import annotations
import argparse
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym

from ik_controller import (
    inverse_kinematics_2link,
    pd_torque,
    joint_angles_from_obs,
    joint_velocities_from_obs,
    target_from_obs,
)
# ---------------------------------------------------------------------------
# Per-step storage helper. Mirrors the lander schema (see DESIGN.md section 4).
# Save is done by save_v2_dataset() below (adds dataset_version=2 metadata).
# ---------------------------------------------------------------------------
class FlatBuffer:
    def __init__(self) -> None:
        self.ep_index:      List[int]        = []
        self.step_index:    List[int]        = []
        self.episode_seed:  List[int]        = []
        self.obs:           List[np.ndarray] = []
        self.actions:       List[np.ndarray] = []
        self.rewards:       List[float]      = []
        self.next_obs:      List[np.ndarray] = []
        self.dones:         List[int]        = []
        self.episode_class: List[int]        = []

    def append(self, ep_id, step, ep_seed, obs, action,
               reward, next_obs, done, klass) -> None:
        self.ep_index.append(int(ep_id))
        self.step_index.append(int(step))
        self.episode_seed.append(int(ep_seed))
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self.dones.append(1 if done else 0)
        self.episode_class.append(int(klass))

    def __len__(self) -> int:
        return len(self.ep_index)


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
ENV_NAME             = "Reacher-v5"
EPISODE_LENGTH       = 100
LINK_LENGTH_1        = 0.10
LINK_LENGTH_2        = 0.11

# v2 episode-class codes (distinct from v1's 0..4 to avoid silent confusion).
CLASS_SMOOTH_IK        = 0
CLASS_AGGRESSIVE_IK    = 1
CLASS_WRONG_TARGET_IK  = 2
CLASS_BLOCK_RANDOM     = 3
CLASS_IID_RANDOM       = 4
CLASS_ZERO             = 5

CLASS_NAMES = {
    CLASS_SMOOTH_IK:        "smooth_ik",
    CLASS_AGGRESSIVE_IK:    "aggressive_ik",
    CLASS_WRONG_TARGET_IK:  "wrong_target_ik",
    CLASS_BLOCK_RANDOM:     "block_random",
    CLASS_IID_RANDOM:       "iid_random",
    CLASS_ZERO:             "zero",
}

DEFAULT_FRACTIONS: Dict[int, float] = {
    CLASS_SMOOTH_IK:       0.35,
    CLASS_AGGRESSIVE_IK:   0.20,
    CLASS_WRONG_TARGET_IK: 0.15,
    CLASS_BLOCK_RANDOM:    0.15,
    CLASS_IID_RANDOM:      0.10,
    CLASS_ZERO:            0.05,
}

# Policy gains.
SMOOTH_KP, SMOOTH_KD, SMOOTH_NOISE             = 2.0, 0.3, 0.05
AGGRESSIVE_KP_DEFAULT                           = 8.0
AGGRESSIVE_KD_DEFAULT                           = 0.0
AGG_NOISE                                       = 0.05
WRONG_KP, WRONG_KD, WRONG_NOISE                 = 2.0, 0.3, 0.05

# Block random: amplitude buckets + fixed block length.
BLOCK_RANDOM_AMPS = (0.3, 0.7, 1.0)
BLOCK_RANDOM_K    = 10

# Wrong-target IK: disk radius for the wrong target sampling.
WRONG_TARGET_DISK_R = 0.21

# Velocity safety filter threshold (rad/s).
DEFAULT_VELOCITY_LIMIT = 80.0

# Curated good/bad return thresholds (printed in summary only).
R_GOOD = -10.0
R_BAD  = -25.0

# CLI defaults.
DEFAULT_TRAIN_EPS    = 1500
DEFAULT_VAL_EPS      = 240
DEFAULT_SEED         = 12345
DEFAULT_TRAIN_PATH   = "reacher_train_dataset.npz"
DEFAULT_VAL_PATH     = "reacher_val_dataset.npz"
DEFAULT_VIEWER_W     = 1024
DEFAULT_VIEWER_H     = 1024

DATASET_VERSION      = 2


# ---------------------------------------------------------------------------
# Velocity safety filter.
# ---------------------------------------------------------------------------
def velocity_safety_filter(action: np.ndarray, obs: np.ndarray,
                           limit: float) -> Tuple[np.ndarray, bool]:
    """If |theta_dot[i]| > limit, force action[i] to oppose joint motion.

    Returns (filtered_action, was_filtered). 'was_filtered' is True if any
    joint hit the safety threshold this step.
    """
    theta_dot = joint_velocities_from_obs(obs)
    out = np.array(action, dtype=np.float32, copy=True)
    triggered = False
    for i in range(2):
        if abs(theta_dot[i]) > limit:
            out[i] = float(-np.sign(theta_dot[i]))
            triggered = True
    return out, triggered


# ---------------------------------------------------------------------------
# Policy classes. Each has reset(env_target, rng) + act(obs, rng).
# velocity_safe flag controls whether the safety filter is applied.
# ---------------------------------------------------------------------------
def _ik_pd_action(obs: np.ndarray, target_xy: np.ndarray,
                  kp: float, kd: float, noise_sigma: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Shared IK + PD + noise + clip helper."""
    theta_now  = joint_angles_from_obs(obs)
    theta_dot  = joint_velocities_from_obs(obs)
    t0, t1 = inverse_kinematics_2link(
        float(target_xy[0]), float(target_xy[1]),
        l1=LINK_LENGTH_1, l2=LINK_LENGTH_2, elbow="up",
    )
    action = pd_torque(
        theta_current=theta_now, theta_dot=theta_dot,
        theta_target=np.array([t0, t1], dtype=np.float64),
        kp=kp, kd=kd, action_low=-1.0, action_high=1.0,
    )
    if noise_sigma > 0.0:
        action = action + rng.normal(0.0, noise_sigma, size=(2,)).astype(np.float32)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
    return action


class Policy:
    """Base policy. Subclasses override act(); reset() is optional."""
    velocity_safe = False

    def reset(self, env_target_xy: np.ndarray, rng: np.random.Generator) -> None:
        pass

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class SmoothIKPolicy(Policy):
    def act(self, obs, rng):
        return _ik_pd_action(obs, target_from_obs(obs),
                             SMOOTH_KP, SMOOTH_KD, SMOOTH_NOISE, rng)


class AggressiveIKPolicy(Policy):
    velocity_safe = True

    def __init__(self, kp: float = AGGRESSIVE_KP_DEFAULT,
                 kd: float = AGGRESSIVE_KD_DEFAULT):
        self.kp = float(kp)
        self.kd = float(kd)

    def act(self, obs, rng):
        return _ik_pd_action(obs, target_from_obs(obs),
                             self.kp, self.kd, AGG_NOISE, rng)


class WrongTargetIKPolicy(Policy):
    def __init__(self, disk_radius: float = WRONG_TARGET_DISK_R):
        self.disk_r = float(disk_radius)
        self.wrong_target = np.zeros(2, dtype=np.float64)

    def reset(self, env_target_xy, rng):
        # Uniform sampling on a disk via rejection. Cheap; disk fills ~78% of box.
        while True:
            x, y = rng.uniform(-self.disk_r, self.disk_r, size=2)
            if x * x + y * y <= self.disk_r * self.disk_r:
                self.wrong_target[0] = x
                self.wrong_target[1] = y
                return

    def act(self, obs, rng):
        return _ik_pd_action(obs, self.wrong_target,
                             WRONG_KP, WRONG_KD, WRONG_NOISE, rng)


class BlockRandomPolicy(Policy):
    velocity_safe = True

    def __init__(self, amps=BLOCK_RANDOM_AMPS, K: int = BLOCK_RANDOM_K):
        self.amps = tuple(amps)
        self.K = int(K)
        self.A = float(amps[0])
        self.steps_in_block = self.K     # force fresh sample on first call
        self.current_action = np.zeros(2, dtype=np.float32)

    def reset(self, env_target_xy, rng):
        self.A = float(rng.choice(self.amps))
        self.steps_in_block = self.K
        self.current_action = np.zeros(2, dtype=np.float32)

    def act(self, obs, rng):
        if self.steps_in_block >= self.K:
            self.current_action = rng.uniform(-self.A, self.A, size=2).astype(np.float32)
            self.steps_in_block = 0
        self.steps_in_block += 1
        return self.current_action.copy()


class IIDRandomPolicy(Policy):
    def act(self, obs, rng):
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class ZeroPolicy(Policy):
    def act(self, obs, rng):
        return np.zeros(2, dtype=np.float32)


def make_policy(klass: int, aggressive_kp: float,
                aggressive_kd: float = AGGRESSIVE_KD_DEFAULT) -> Policy:
    if klass == CLASS_SMOOTH_IK:
        return SmoothIKPolicy()
    if klass == CLASS_AGGRESSIVE_IK:
        return AggressiveIKPolicy(kp=aggressive_kp, kd=aggressive_kd)
    if klass == CLASS_WRONG_TARGET_IK:
        return WrongTargetIKPolicy()
    if klass == CLASS_BLOCK_RANDOM:
        return BlockRandomPolicy()
    if klass == CLASS_IID_RANDOM:
        return IIDRandomPolicy()
    if klass == CLASS_ZERO:
        return ZeroPolicy()
    raise ValueError(f"Unknown class: {klass}")


# ---------------------------------------------------------------------------
# Schedule generation.
# ---------------------------------------------------------------------------
def make_class_schedule(num_episodes: int,
                        fractions: Dict[int, float],
                        rng: np.random.Generator) -> np.ndarray:
    """Return shape (num_episodes,) shuffled schedule of class codes.
    Bucket counts are rounded; the largest bucket absorbs the remainder."""
    classes = sorted(fractions.keys())
    counts: Dict[int, int] = {c: int(round(fractions[c] * num_episodes)) for c in classes}
    diff = num_episodes - sum(counts.values())
    biggest = max(classes, key=lambda c: counts[c])
    counts[biggest] += diff

    parts = [np.full(counts[c], c, dtype=np.int64) for c in classes if counts[c] > 0]
    schedule = np.concatenate(parts)
    rng.shuffle(schedule)
    return schedule


# ---------------------------------------------------------------------------
# Episode runner.
# ---------------------------------------------------------------------------
def run_episode(env, ep_id: int, ep_seed: int, klass: int,
                policy: Policy, rng: np.random.Generator,
                buf: FlatBuffer, velocity_limit: float) -> Dict[str, float]:
    """Run one episode and append every step to buf. Returns per-episode stats."""
    obs, _ = env.reset(seed=ep_seed)
    env_target = obs[4:6].copy()
    policy.reset(env_target, rng)

    ep_return = 0.0
    steps = 0
    sat_steps = 0
    vsafety_steps = 0
    max_speed = 0.0

    for step in range(EPISODE_LENGTH):
        action = policy.act(obs, rng).astype(np.float32, copy=False)

        if policy.velocity_safe:
            action, triggered = velocity_safety_filter(action, obs, velocity_limit)
            vsafety_steps += int(triggered)

        sat_steps += int(np.any(np.abs(action) > 0.99))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buf.append(ep_id, step, ep_seed, obs, action, reward, next_obs, done, klass)

        ep_return += float(reward)
        steps += 1
        max_speed = max(max_speed, float(max(abs(next_obs[6]), abs(next_obs[7]))))
        obs = next_obs
        if done:
            break

    return {
        "return":    ep_return,
        "steps":     steps,
        "sat_pct":   100.0 * sat_steps / max(steps, 1),
        "vsafe_pct": 100.0 * vsafety_steps / max(steps, 1),
        "max_speed": max_speed,
    }


# ---------------------------------------------------------------------------
# NPZ save helper -- preserves FlatBuffer schema and adds dataset_version.
# ---------------------------------------------------------------------------
def save_v2_dataset(buf: FlatBuffer, path: str) -> None:
    if len(buf) == 0:
        print(f"  [skip] nothing to save in {path}")
        return
    np.savez(
        path,
        ep_index        = np.asarray(buf.ep_index,      dtype=np.int64),
        step_index      = np.asarray(buf.step_index,    dtype=np.int64),
        episode_seed    = np.asarray(buf.episode_seed,  dtype=np.int64),
        obs             = np.stack(buf.obs,             axis=0).astype(np.float32),
        actions         = np.stack(buf.actions,         axis=0).astype(np.float32),
        rewards         = np.asarray(buf.rewards,       dtype=np.float32),
        next_obs        = np.stack(buf.next_obs,        axis=0).astype(np.float32),
        dones           = np.asarray(buf.dones,         dtype=np.int64),
        episode_class   = np.asarray(buf.episode_class, dtype=np.int64),
        dataset_version = np.int64(DATASET_VERSION),
    )
    print(f"  saved {len(buf)} transitions to {path}")


# ---------------------------------------------------------------------------
# Per-set collection.
# ---------------------------------------------------------------------------
def collect_set(set_name: str, num_episodes: int, base_seed: int,
                output_path: str, headless: bool, ep_id_offset: int,
                aggressive_kp: float, aggressive_kd: float,
                velocity_limit: float,
                fractions: Dict[int, float],
                viewer_w: int = DEFAULT_VIEWER_W,
                viewer_h: int = DEFAULT_VIEWER_H) -> List[Tuple[int, float]]:
    print(f"\n=== Collecting {set_name}: {num_episodes} episodes "
          f"(render={'headless' if headless else f'human {viewer_w}x{viewer_h}'}) ===")

    if headless:
        env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=EPISODE_LENGTH)
    else:
        env = gym.make(ENV_NAME, render_mode="human",
                       max_episode_steps=EPISODE_LENGTH,
                       width=viewer_w, height=viewer_h)

    class_rng  = np.random.default_rng(base_seed)
    seed_rng   = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    action_rng = np.random.default_rng(base_seed ^ 0x5A5A5A5A)

    schedule = make_class_schedule(num_episodes, fractions, class_rng)
    bucket_summary = "  bucket counts: " + "  ".join(
        f"{CLASS_NAMES[c]}={int((schedule == c).sum())}"
        for c in sorted(set(int(x) for x in schedule))
    )
    print(bucket_summary)

    buf = FlatBuffer()
    ep_summaries: List[Tuple[int, float]] = []
    per_class_stats: Dict[int, Dict[str, List[float]]] = {
        c: {"return": [], "sat_pct": [], "vsafe_pct": [], "max_speed": []}
        for c in sorted(set(int(x) for x in schedule))
    }

    t_start = time.time()
    try:
        for i in range(num_episodes):
            klass = int(schedule[i])
            ep_seed = int(seed_rng.integers(0, 2**31 - 1))
            ep_id = ep_id_offset + i

            policy = make_policy(klass, aggressive_kp, aggressive_kd)
            stats = run_episode(env, ep_id, ep_seed, klass, policy,
                                action_rng, buf, velocity_limit)

            ep_summaries.append((klass, stats["return"]))
            per_class_stats[klass]["return"].append(stats["return"])
            per_class_stats[klass]["sat_pct"].append(stats["sat_pct"])
            per_class_stats[klass]["vsafe_pct"].append(stats["vsafe_pct"])
            per_class_stats[klass]["max_speed"].append(stats["max_speed"])

            if (i + 1) % 50 == 0 or i == num_episodes - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta  = (num_episodes - i - 1) / max(rate, 1e-6)
                print(f"  ep {i+1:>4}/{num_episodes}  "
                      f"class={CLASS_NAMES[klass]:<16}  "
                      f"return={stats['return']:+8.2f}  "
                      f"sat%={stats['sat_pct']:5.1f}  "
                      f"vsafe%={stats['vsafe_pct']:5.1f}  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] saving partial dataset...")
    finally:
        env.close()

    save_v2_dataset(buf, output_path)

    # ---- Per-class stats summary ----
    print(f"\n  --- {set_name} per-class stats ---")
    print(f"  {'class':<16}  {'n':>4}  {'mean_ret':>9}  {'min_ret':>9}  "
          f"{'max_ret':>9}  {'sat%':>5}  {'vsafe%':>6}  {'max_spd':>7}")
    for c in sorted(per_class_stats):
        rs = np.asarray(per_class_stats[c]["return"], dtype=np.float64)
        sat = np.asarray(per_class_stats[c]["sat_pct"], dtype=np.float64)
        vs  = np.asarray(per_class_stats[c]["vsafe_pct"], dtype=np.float64)
        sp  = np.asarray(per_class_stats[c]["max_speed"], dtype=np.float64)
        if rs.size == 0:
            continue
        print(f"  {CLASS_NAMES[c]:<16}  {rs.size:>4}  "
              f"{rs.mean():+9.2f}  {rs.min():+9.2f}  {rs.max():+9.2f}  "
              f"{sat.mean():5.1f}  {vs.mean():6.1f}  {sp.max():7.1f}")

    return ep_summaries


# ---------------------------------------------------------------------------
# Summary printer (re-used from v1 conceptually).
# ---------------------------------------------------------------------------
def print_return_summary(set_name: str, summaries: List[Tuple[int, float]]) -> None:
    if not summaries:
        return
    print(f"\n--- {set_name} return summary ({len(summaries)} episodes) ---")
    rs_all = np.array([r for _, r in summaries], dtype=np.float64)
    n_good = int((rs_all >= R_GOOD).sum())
    n_bad  = int((rs_all <= R_BAD).sum())
    n_mid  = len(rs_all) - n_good - n_bad
    print(f"  curated split  (R_good>={R_GOOD:+.1f}, R_bad<={R_BAD:+.1f}): "
          f"good={n_good}  mid={n_mid}  bad={n_bad}")
    bins = [-100, -90, -80, -70, -60, -50, -40, -30, -25, -20, -15, -10, -7, -5, -3, -1, 0]
    counts, edges = np.histogram(rs_all, bins=bins)
    print("  histogram:")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(50 * c / max(counts.max(), 1))
        print(f"    [{lo:+6.1f}, {hi:+6.1f})  {c:>4}  {bar}")


# ---------------------------------------------------------------------------
# Calibration mode for aggressive_ik kp / kd.
# ---------------------------------------------------------------------------
def _acf1(a: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D sequence (NaN-safe)."""
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    denom = float(np.dot(a, a))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a[:-1], a[1:]) / denom)


def _run_one_calibration(env, kp: float, kd: float,
                         episodes: int, velocity_limit: float,
                         seed_rng, action_rng) -> Dict[str, float]:
    """Run `episodes` aggressive_ik episodes at the given (kp, kd) and return
    aggregate stats including mean ACF(1) of the action sequence (lower is
    smoother; strongly negative ⇒ chattering)."""
    rets: List[float] = []
    sats: List[float] = []
    vsafes: List[float] = []
    speeds: List[float] = []
    acf0_list: List[float] = []
    acf1_list: List[float] = []
    runs_ge10 = 0
    for _ in range(episodes):
        ep_seed = int(seed_rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        policy = AggressiveIKPolicy(kp=kp, kd=kd)
        policy.reset(obs[4:6].copy(), action_rng)
        ep_ret = 0.0
        n_sat_step = 0
        n_vs_step = 0
        max_spd = 0.0
        run = best_run = 0
        acts: List[np.ndarray] = []
        for _t in range(EPISODE_LENGTH):
            action = policy.act(obs, action_rng).astype(np.float32, copy=False)
            action, triggered = velocity_safety_filter(action, obs, velocity_limit)
            n_vs_step += int(triggered)
            acts.append(action.copy())
            hit_boundary = bool(np.any(np.abs(action) >= 0.9))
            if hit_boundary:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
            n_sat_step += int(np.any(np.abs(action) > 0.99))
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            max_spd = max(max_spd, float(max(abs(obs[6]), abs(obs[7]))))
            if term or trunc:
                break
        rets.append(ep_ret)
        sats.append(100.0 * n_sat_step / EPISODE_LENGTH)
        vsafes.append(100.0 * n_vs_step / EPISODE_LENGTH)
        speeds.append(max_spd)
        if best_run >= 10:
            runs_ge10 += 1
        A = np.asarray(acts, dtype=np.float64)
        if A.shape[0] >= 3:
            acf0_list.append(_acf1(A[:, 0]))
            acf1_list.append(_acf1(A[:, 1]))

    rs = np.asarray(rets)
    ss = np.asarray(sats)
    vs = np.asarray(vsafes)
    sp = np.asarray(speeds)
    acf_all = np.asarray([a for a in (acf0_list + acf1_list) if not np.isnan(a)])
    return {
        "n":         float(len(rs)),
        "mean_ret":  float(rs.mean()) if rs.size else float("nan"),
        "mean_sat":  float(ss.mean()) if ss.size else float("nan"),
        "mean_vs":   float(vs.mean()) if vs.size else float("nan"),
        "max_spd":   float(sp.max())  if sp.size else float("nan"),
        "runs_ge10": float(runs_ge10),
        "mean_acf1": float(acf_all.mean()) if acf_all.size else float("nan"),
    }


def _print_calibration_row(axis_name: str, axis_val: float,
                           kp: float, kd: float, s: Dict[str, float]) -> None:
    print(f"  {axis_val:>5.2f}  kp={kp:>4.1f} kd={kd:>4.2f}  "
          f"{int(s['n']):>3}  "
          f"{s['mean_ret']:+9.2f}  "
          f"{s['mean_sat']:>10.1f}  "
          f"{s['mean_vs']:>12.1f}  "
          f"{s['max_spd']:>9.1f}  "
          f"{int(s['runs_ge10']):>18d}  "
          f"{s['mean_acf1']:>+8.3f}")


def run_calibration_kp(seed: int, kps: List[float], fixed_kd: float,
                       velocity_limit: float, episodes_per_kp: int) -> None:
    print(f"\n=== --calibrate_kp: testing aggressive_ik with "
          f"{episodes_per_kp} episodes per kp in {kps} (kd={fixed_kd}) ===\n")
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=EPISODE_LENGTH)
    seed_rng   = np.random.default_rng(seed ^ 0xA5A5A5A5)
    action_rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    print(f"  {'kp':>5}  {'gains':>14}  {'n':>3}  {'mean_ret':>9}  "
          f"{'mean_sat%':>10}  {'mean_vsafe%':>12}  {'max_speed':>9}  "
          f"{'run>=10 @|a|>=0.9':>18}  {'ACF(1)':>8}")
    for kp in kps:
        stats = _run_one_calibration(env, kp, fixed_kd, episodes_per_kp,
                                     velocity_limit, seed_rng, action_rng)
        _print_calibration_row("kp", kp, kp, fixed_kd, stats)
    env.close()
    print("\n  Pick a kp that gives sat% > 10, max_speed < 120 rad/s, "
          "and ACF(1) > 0 (negative ACF means chattering).")


def run_calibration_kd(seed: int, kds: List[float], fixed_kp: float,
                       velocity_limit: float, episodes_per_kd: int) -> None:
    print(f"\n=== --calibrate_kd: testing aggressive_ik with "
          f"{episodes_per_kd} episodes per kd in {kds} (kp={fixed_kp}) ===\n")
    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=EPISODE_LENGTH)
    seed_rng   = np.random.default_rng(seed ^ 0xA5A5A5A5)
    action_rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    print(f"  {'kd':>5}  {'gains':>14}  {'n':>3}  {'mean_ret':>9}  "
          f"{'mean_sat%':>10}  {'mean_vsafe%':>12}  {'max_speed':>9}  "
          f"{'run>=10 @|a|>=0.9':>18}  {'ACF(1)':>8}")
    for kd in kds:
        stats = _run_one_calibration(env, fixed_kp, kd, episodes_per_kd,
                                     velocity_limit, seed_rng, action_rng)
        _print_calibration_row("kd", kd, fixed_kp, kd, stats)
    env.close()
    print("\n  Pick a kd that keeps ACF(1) >= +0.3 (sustained, smooth boundary "
          "actions) while preserving sat% and run>=10 count.")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Reacher-v5 v2 dataset (train + val, 6 buckets)."
    )
    parser.add_argument("--train_episodes", type=int, default=DEFAULT_TRAIN_EPS)
    parser.add_argument("--val_episodes",   type=int, default=DEFAULT_VAL_EPS)
    parser.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_path",     default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path",       default=DEFAULT_VAL_PATH)
    parser.add_argument("--headless",       action="store_true")
    parser.add_argument("--viewer_width",   type=int, default=DEFAULT_VIEWER_W)
    parser.add_argument("--viewer_height",  type=int, default=DEFAULT_VIEWER_H)
    parser.add_argument("--skip_train",     action="store_true")
    parser.add_argument("--skip_val",       action="store_true")

    parser.add_argument("--aggressive_kp",  type=float, default=AGGRESSIVE_KP_DEFAULT,
                        help=f"P-gain for aggressive_ik bucket (default {AGGRESSIVE_KP_DEFAULT})")
    parser.add_argument("--aggressive_kd",  type=float, default=AGGRESSIVE_KD_DEFAULT,
                        help=f"D-gain for aggressive_ik bucket (default {AGGRESSIVE_KD_DEFAULT})")
    parser.add_argument("--velocity_limit", type=float, default=DEFAULT_VELOCITY_LIMIT,
                        help=f"|theta_dot| threshold for safety filter rad/s "
                             f"(default {DEFAULT_VELOCITY_LIMIT})")

    parser.add_argument("--calibrate_kp",   action="store_true",
                        help="Sweep aggressive_ik kp over --calibrate_kps at fixed "
                             "--aggressive_kd. Does NOT collect a dataset.")
    parser.add_argument("--calibrate_kps",  type=float, nargs="+",
                        default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    parser.add_argument("--calibrate_kd",   action="store_true",
                        help="Sweep aggressive_ik kd over --calibrate_kds at fixed "
                             "--aggressive_kp. Does NOT collect a dataset.")
    parser.add_argument("--calibrate_kds",  type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--calibrate_eps",  type=int, default=5,
                        help="Episodes per (kp or kd) during calibration (default 5)")

    args = parser.parse_args()

    if args.calibrate_kp and args.calibrate_kd:
        print("error: cannot pass --calibrate_kp and --calibrate_kd together",
              file=sys.stderr)
        return 2

    if args.calibrate_kp:
        run_calibration_kp(args.seed, args.calibrate_kps, args.aggressive_kd,
                           args.velocity_limit, args.calibrate_eps)
        return 0

    if args.calibrate_kd:
        run_calibration_kd(args.seed, args.calibrate_kds, args.aggressive_kp,
                           args.velocity_limit, args.calibrate_eps)
        return 0

    print(f"Reacher-v5 v2 dataset collection")
    print(f"  env={ENV_NAME}  episode_length={EPISODE_LENGTH}")
    print(f"  buckets:")
    for c in sorted(DEFAULT_FRACTIONS):
        print(f"    {CLASS_NAMES[c]:<16} {DEFAULT_FRACTIONS[c]*100:5.1f}%")
    print(f"  aggressive_kp = {args.aggressive_kp}")
    print(f"  aggressive_kd = {args.aggressive_kd}")
    print(f"  velocity_limit = {args.velocity_limit} rad/s")
    print(f"  master seed = {args.seed}")

    train_summaries: List[Tuple[int, float]] = []
    val_summaries:   List[Tuple[int, float]] = []

    if not args.skip_train:
        train_summaries = collect_set(
            set_name="train", num_episodes=args.train_episodes,
            base_seed=args.seed, output_path=args.train_path,
            headless=args.headless, ep_id_offset=0,
            aggressive_kp=args.aggressive_kp,
            aggressive_kd=args.aggressive_kd,
            velocity_limit=args.velocity_limit,
            fractions=DEFAULT_FRACTIONS,
            viewer_w=args.viewer_width, viewer_h=args.viewer_height,
        )
    if not args.skip_val:
        val_summaries = collect_set(
            set_name="val", num_episodes=args.val_episodes,
            base_seed=args.seed + 1_000_000,
            output_path=args.val_path,
            headless=args.headless,
            ep_id_offset=args.train_episodes,
            aggressive_kp=args.aggressive_kp,
            aggressive_kd=args.aggressive_kd,
            velocity_limit=args.velocity_limit,
            fractions=DEFAULT_FRACTIONS,
            viewer_w=args.viewer_width, viewer_h=args.viewer_height,
        )

    print_return_summary("train", train_summaries)
    print_return_summary("val",   val_summaries)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
