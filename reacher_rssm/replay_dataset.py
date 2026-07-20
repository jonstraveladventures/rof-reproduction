# ===========================================================================
# Reacher RSSM - dataset visual replay.
#
# Loads a .npz dataset produced by collect_dataset.py and replays a few
# episodes in a MuJoCo viewer so the IK / noisy-IK / random policies can be
# eyeballed for sanity. Episodes are reproduced exactly by re-seeding the env
# with the saved per-episode `episode_seed` and feeding back the saved actions.
#
# CLI:
#   python replay_dataset.py --dataset reacher_train_dataset.npz --episodes 10
#   python replay_dataset.py --dataset reacher_val_dataset.npz   --episodes 5 --class good_ik
# ===========================================================================

from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import gymnasium as gym


ENV_NAME       = "Reacher-v5"
EPISODE_LENGTH = 100

CLASS_NAME_TO_INT = {"good_ik": 0, "noisy_ik": 1, "random": 2,
                     "zero": 3, "const": 4, "any": -1}
CLASS_INT_TO_NAME = {0: "good_ik", 1: "noisy_ik", 2: "random",
                     3: "zero", 4: "const"}

# MuJoCo viewer window size (default 480x480 is too small to see the arm).
DEFAULT_VIEWER_W = 1024
DEFAULT_VIEWER_H = 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay episodes from a Reacher dataset .npz")
    parser.add_argument("--dataset",  default="reacher_train_dataset.npz",
                        help="path to dataset .npz (default reacher_train_dataset.npz)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes to replay (default 10)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="RNG seed for picking which episodes to replay")
    parser.add_argument("--class", dest="class_filter", default="any",
                        choices=list(CLASS_NAME_TO_INT.keys()),
                        help="only replay episodes of this class (default any)")
    parser.add_argument("--viewer_width",  type=int, default=DEFAULT_VIEWER_W,
                        help=f"viewer window width in pixels (default {DEFAULT_VIEWER_W})")
    parser.add_argument("--viewer_height", type=int, default=DEFAULT_VIEWER_H,
                        help=f"viewer window height in pixels (default {DEFAULT_VIEWER_H})")
    args = parser.parse_args()

    data = np.load(args.dataset)
    required = ["ep_index", "step_index", "episode_seed", "actions",
                "rewards", "episode_class"]
    for k in required:
        if k not in data.files:
            print(f"  ERROR: dataset missing key '{k}'. Did you save it with the new schema?")
            return 1

    ep_index      = data["ep_index"].astype(np.int64)
    step_index    = data["step_index"].astype(np.int64)
    episode_seed  = data["episode_seed"].astype(np.int64)
    actions       = data["actions"].astype(np.float32)
    rewards       = data["rewards"].astype(np.float32)
    episode_class = data["episode_class"].astype(np.int64)

    if ep_index.size == 0:
        print("  ERROR: empty dataset.")
        return 1

    # Build a per-episode summary in one pass (for reporting + filtering).
    unique_eps = np.unique(ep_index)
    ep_meta = []
    for ep_id in unique_eps:
        m = (ep_index == ep_id)
        ep_meta.append({
            "ep_id":   int(ep_id),
            "seed":    int(episode_seed[m][0]),
            "class":   int(episode_class[m][0]),
            "return":  float(rewards[m].sum()),
            "steps":   int(m.sum()),
        })

    # Apply class filter.
    class_filter_int = CLASS_NAME_TO_INT[args.class_filter]
    if class_filter_int >= 0:
        ep_meta = [e for e in ep_meta if e["class"] == class_filter_int]

    if not ep_meta:
        print(f"  ERROR: no episodes of class '{args.class_filter}' in {args.dataset}")
        return 1

    rng = np.random.default_rng(args.seed)
    n_pick = min(args.episodes, len(ep_meta))
    picks = rng.choice(len(ep_meta), size=n_pick, replace=False)

    print(f"Replaying {n_pick} episodes from {args.dataset}")
    print(f"  total episodes in dataset: {len(unique_eps)}  "
          f"(class filter: {args.class_filter})")

    env = gym.make(ENV_NAME, render_mode="human",
                   max_episode_steps=EPISODE_LENGTH,
                   width=args.viewer_width, height=args.viewer_height)
    metadata_fps = env.metadata.get("render_fps", 50)

    try:
        for i, idx in enumerate(picks, start=1):
            meta = ep_meta[int(idx)]
            ep_id, ep_seed, klass = meta["ep_id"], meta["seed"], meta["class"]

            m = (ep_index == ep_id)
            order = np.argsort(step_index[m], kind="stable")
            ep_actions = actions[m][order]

            obs, _ = env.reset(seed=ep_seed)
            replayed_return = 0.0
            for step, a in enumerate(ep_actions):
                obs, r, terminated, truncated, _ = env.step(a)
                replayed_return += float(r)
                env.render()
                if terminated or truncated:
                    break
                # MuJoCo human renderer normally throttles itself; this is a belt-and-braces
                # sleep in case of headless replay (it should be a no-op for human render).
                # Comment out for max-speed replay.
                # time.sleep(1.0 / metadata_fps)

            stored_return = meta["return"]
            drift = abs(stored_return - replayed_return)
            print(f"  [{i:>3}/{n_pick}]  ep_id={ep_id:>4}  "
                  f"class={CLASS_INT_TO_NAME.get(klass, '?'):<8}  "
                  f"steps={meta['steps']}  "
                  f"return(stored)={stored_return:+8.2f}  "
                  f"return(replayed)={replayed_return:+8.2f}  "
                  f"drift={drift:.4f}")
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] stopping replay.")
    finally:
        env.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
