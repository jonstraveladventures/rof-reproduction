"""
Reward-Markovianity diagnostic (paper Section 4.8 protocol).

Trains MLP regressors r_t ~ f(inputs) in four configurations and reports
validation R^2:
  A: (o_t, a_t)
  B: (o_t, a_t), terminal steps filtered out
  C: (o_t, a_t, o_{t+1})
  D: (o_t, a_t, o_{t+1}), terminal steps filtered out

Discrete-paper reference values: A=0.290, B=0.670, C=0.595, D=0.971.
Works for both discrete (int actions -> one-hot) and continuous (float) datasets.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn


def load(path, action_dim):
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    nxt = d["next_obs"].astype(np.float32)
    rew = d["rewards"].astype(np.float32)
    done = d["dones"].astype(np.int64)
    act = d["actions"]
    if act.dtype.kind in "iu":
        a = np.zeros((len(act), action_dim), dtype=np.float32)
        a[np.arange(len(act)), act] = 1.0
    else:
        a = act.astype(np.float32)
    return obs, a, nxt, rew, done


def r2(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def run_config(name, xtr, ytr, xva, yva, epochs, device, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(xtr.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).unsqueeze(1).to(device)
    n = len(xtr_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = ((model(xtr_t[idx]) - ytr_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(torch.from_numpy(xva).to(device)).squeeze(1).cpu().numpy()
    score = r2(pred, yva)
    print(f"  {name}: R^2 = {score:.3f}")
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dataset", required=True)
    p.add_argument("--val_dataset", required=True)
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    otr, atr, ntr, rtr, dtr = load(args.train_dataset, args.action_dim)
    ova, ava, nva, rva, dva = load(args.val_dataset, args.action_dim)
    print(f"train {len(otr)} steps, val {len(ova)} steps, device={device}")
    ktr, kva = dtr == 0, dva == 0
    print(f"terminal steps filtered in B/D: train {np.sum(~ktr)}, val {np.sum(~kva)}")

    cfgs = {
        "A (o,a)          ": (np.hstack([otr, atr]), rtr, np.hstack([ova, ava]), rva),
        "B (o,a) no-term  ": (np.hstack([otr, atr])[ktr], rtr[ktr], np.hstack([ova, ava])[kva], rva[kva]),
        "C (o,a,o')       ": (np.hstack([otr, atr, ntr]), rtr, np.hstack([ova, ava, nva]), rva),
        "D (o,a,o') no-term": (np.hstack([otr, atr, ntr])[ktr], rtr[ktr], np.hstack([ova, ava, nva])[kva], rva[kva]),
    }
    print("validation R^2 by configuration:")
    for name, (xtr, ytr, xva, yva) in cfgs.items():
        run_config(name, xtr, ytr, xva, yva, args.epochs, device, args.seed)


if __name__ == "__main__":
    main()
