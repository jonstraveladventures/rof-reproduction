"""
Build the data-quality-dial datasets (pre-registered 2026-07-07).

Episode-level mixing of the author's human demonstrations with our scripted
demonstrations, plus human-subset cells. Composition seed 12345, sealed.

Train conditions (750 episodes each unless a subset):
  mix_f25: 188 human + 562 scripted     mix_f50: 375 + 375
  mix_f75: 562 human + 188 scripted
  human_sub25: 188 human only           human_sub50: 375 human only
Val conditions (122 episodes, mixed at the same f; subsets use full human val).
"""

import numpy as np

RNG = np.random.default_rng(12345)
KEYS = ("obs", "next_obs", "rewards", "dones", "ep_index", "episode_seed", "step_index", "actions")


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in KEYS}


def episodes(data):
    ids = np.unique(data["ep_index"])
    return [{k: data[k][data["ep_index"] == i] for k in KEYS} for i in ids]


def assemble(ep_list, out_path):
    cols = {k: [] for k in KEYS}
    for new_id, ep in enumerate(ep_list):
        for k in KEYS:
            if k == "ep_index":
                cols[k].append(np.full(len(ep[k]), new_id, dtype=np.int64))
            else:
                cols[k].append(ep[k])
    out = {k: np.concatenate(cols[k]) for k in KEYS}
    np.savez_compressed(out_path, **out)
    rets = np.array([ep["rewards"].sum() for ep in ep_list])
    good, bad = (rets >= 100).mean(), (rets <= -100).mean()
    print(f"{out_path}: {len(ep_list)} eps, {len(out['obs'])} steps, "
          f"mix {good:.1%}/{bad:.1%}/{1-good-bad:.1%}")


def main():
    h_tr = episodes(load("lunarlander_train_dataset.npz"))
    s_tr = episodes(load("lunarlander_scripted_train_dataset.npz"))
    h_va = episodes(load("lunarlander_val_dataset.npz"))
    s_va = episodes(load("lunarlander_scripted_val_dataset.npz"))
    print(f"pools: human {len(h_tr)}/{len(h_va)}, scripted {len(s_tr)}/{len(s_va)}")

    # fixed random orderings, drawn once (sealed seed)
    h_tr_ord = RNG.permutation(len(h_tr))
    s_tr_ord = RNG.permutation(len(s_tr))
    h_va_ord = RNG.permutation(len(h_va))
    s_va_ord = RNG.permutation(len(s_va))

    for f, tag in ((0.25, "f25"), (0.5, "f50"), (0.75, "f75")):
        n_h_tr = round(750 * f)
        n_h_va = round(122 * f)
        tr = [h_tr[i] for i in h_tr_ord[:n_h_tr]] + [s_tr[i] for i in s_tr_ord[:750 - n_h_tr]]
        va = [h_va[i] for i in h_va_ord[:n_h_va]] + [s_va[i] for i in s_va_ord[:122 - n_h_va]]
        assemble(tr, f"mix_{tag}_train.npz")
        assemble(va, f"mix_{tag}_val.npz")

    for frac, tag in ((0.25, "sub25"), (0.5, "sub50")):
        n = round(750 * frac)
        assemble([h_tr[i] for i in h_tr_ord[:n]], f"human_{tag}_train.npz")

    print("done (subsets use the full human val set)")


if __name__ == "__main__":
    main()
