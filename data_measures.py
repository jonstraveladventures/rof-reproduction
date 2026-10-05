"""
Dataset measures for the data-quality dial, defined on 2026-07-07 and
computed from data alone, before any training outcome existed.

M1 coverage: mean log distance to the 5th nearest neighbour over 20k
   subsampled (z-scored obs ++ one-hot action) points.
M2 local action diversity: median over 5k sampled states of the categorical
   entropy of the action histogram among the state's 10 nearest neighbours.
M3 episode non-redundancy: median pairwise distance between episode
   signatures (10 evenly spaced obs waypoints, flattened, z-scored) over
   300 random episode pairs.
M4 return-mix distance to the human mix (L1 on good/bad/middle).
M6 size (episodes, steps). [M5 Markovianity runs separately.]
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(12345)
HUMAN_MIX = None


def episode_slices(d):
    ids = np.unique(d["ep_index"])
    return [np.where(d["ep_index"] == i)[0] for i in ids]


def measures(path):
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float64)
    act = d["actions"].astype(np.int64)
    n_act = 4
    mu, sd = obs.mean(0), obs.std(0) + 1e-8
    z = (obs - mu) / sd

    # M1: coverage
    idx = RNG.choice(len(z), size=min(20000, len(z)), replace=False)
    pts = np.hstack([z[idx], np.eye(n_act)[act[idx]]])
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=6)
    m1 = float(np.mean(np.log(dist[:, 5] + 1e-12)))

    # M2: local action diversity
    tree_s = cKDTree(z[idx])
    q = RNG.choice(len(idx), size=min(5000, len(idx)), replace=False)
    _, nn = tree_s.query(z[idx][q], k=10)
    ents = []
    for row in nn:
        counts = np.bincount(act[idx][row], minlength=n_act).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        ents.append(float(-(p * np.log(p)).sum()))
    m2 = float(np.median(ents))

    # M3: episode non-redundancy
    slices = episode_slices(d)
    sigs = []
    for sl in slices:
        w = np.linspace(0, len(sl) - 1, 10).astype(int)
        sigs.append(z[sl[w]].ravel())
    sigs = np.array(sigs)
    pairs = RNG.integers(0, len(sigs), size=(300, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    m3 = float(np.median(np.linalg.norm(sigs[pairs[:, 0]] - sigs[pairs[:, 1]], axis=1)))

    # M4: return-mix distance to human
    rets = np.array([d["rewards"][sl].sum() for sl in slices])
    mix = np.array([(rets >= 100).mean(), (rets <= -100).mean()])
    mix = np.append(mix, 1 - mix.sum())
    global HUMAN_MIX
    if HUMAN_MIX is None and "lunarlander_train_dataset.npz" in path:
        HUMAN_MIX = mix
    m4 = float(np.abs(mix - HUMAN_MIX).sum()) if HUMAN_MIX is not None else float("nan")

    return m1, m2, m3, m4, len(slices), len(obs)


if __name__ == "__main__":
    files = ["lunarlander_train_dataset.npz",       # f=1 human (first: sets HUMAN_MIX)
             "lunarlander_scripted_train_dataset.npz",  # f=0
             "mix_f25_train.npz", "mix_f50_train.npz", "mix_f75_train.npz",
             "human_sub25_train.npz", "human_sub50_train.npz"]
    print(f"{'dataset':38} {'M1_cov':>8} {'M2_actdiv':>9} {'M3_epdist':>9} {'M4_mixL1':>8} {'eps':>5} {'steps':>7}")
    for f in files:
        m1, m2, m3, m4, ne, ns = measures(f)
        print(f"{f:38} {m1:>8.3f} {m2:>9.3f} {m3:>9.3f} {m4:>8.3f} {ne:>5} {ns:>7}")
