#!/usr/bin/env python3
"""Planner-side (edge-of-reach) mechanism test.

Design and sealed predictions: pre-registrations/2026-09-04-edge-of-reach.md.
For each checkpoint and each pre-chosen real start state (val episode, pre-contact step),
warm up the belief on real observations, plan with the MPC-eval CEM (eval defaults), then
compare the imagined rollout of the chosen sequence with its real execution, alongside the
recorded human sequence from the same state. Also measures how far imagined observations
sit from the training-set support (5-NN standardised distance).

Usage:
    python eval_edge.py --run sp909 --checkpoints_dir /path/ckpt_sp909 --output results/edge_sp909.txt
    python eval_edge.py --run v7 --checkpoints_dir wm_checkpoints/v7_12345_continuous --smoke
"""
import argparse, os, re, random, glob, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from models import WorldModel

DEVICE = torch.device("cpu")
H = 25          # planning horizon (= MPC eval)
W = 5           # posterior warmup steps (= metrics)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    wm = cfg.get("world_model", {}); cap = wm.get("capacity", {})
    return (int(cap.get("latent_dim", 16)), int(cap.get("hidden_dim", 256)),
            int(cap.get("mlp_hidden_dim", cap.get("hidden_dim", 256))),
            int(cap.get("gru_num_layers", 1)), int(wm.get("action_dim", 4)))


def load_world_model(config_path, ckpt_path, obs_dim):
    latent_dim, hidden_dim, mlp_hidden_dim, gru_layers, action_dim = load_config(config_path)
    wm = WorldModel(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim,
                    hidden_dim=hidden_dim, gru_num_layers=gru_layers,
                    mlp_hidden_dim=mlp_hidden_dim).to(DEVICE)
    payload = torch.load(ckpt_path, map_location=DEVICE)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    wm.load_state_dict(sd)
    wm.eval()
    return wm, action_dim


class CEM:  # MPC-eval defaults (wm_mpc_policy.py argparse defaults; obs cost off)
    horizon = H; population = 384; elites = 48; cem_iters = 4; cem_alpha = 0.7
    temperature = 1.0; gamma = 0.97; reward_weight = 0.2; done_penalty = 2.5


@torch.no_grad()
def score_sequences(wm, h0, z0, seqs, action_dim):
    B = seqs.shape[0]
    h = h0.expand(B, -1).contiguous() if h0.dim() == 2 else h0.expand(B, -1, -1).contiguous()
    z = z0.expand(B, -1).contiguous()
    scores = torch.zeros(B, device=DEVICE); disc = 1.0
    for t in range(seqs.shape[1]):
        a = F.one_hot(seqs[:, t], num_classes=action_dim).float()
        h = wm.rssm.update_hidden(h, z, a)
        z, _ = wm.rssm.prior(h)
        r = wm.predict_reward(h, z)
        s = CEM.reward_weight * r
        if CEM.done_penalty > 0.0 and hasattr(wm, "predict_done_logits"):
            s = s - CEM.done_penalty * torch.sigmoid(wm.predict_done_logits(h, z))
        scores = scores + disc * s; disc *= CEM.gamma
    return scores


@torch.no_grad()
def cem_plan(wm, h0, z0, action_dim):
    logits = torch.zeros(CEM.horizon, action_dim, device=DEVICE)
    best_seq, best_score = None, None
    for _ in range(CEM.cem_iters):
        probs = torch.softmax(logits / max(CEM.temperature, 1e-6), dim=-1)
        dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(CEM.population, -1, -1))
        seqs = dist.sample()
        scores = score_sequences(wm, h0, z0, seqs, action_dim)
        top, idx = torch.max(scores, dim=0)
        if best_score is None or top.item() > best_score:
            best_score = float(top.item()); best_seq = seqs[idx].clone()
        elite = seqs[torch.topk(scores, k=min(CEM.elites, CEM.population), dim=0).indices]
        freq = F.one_hot(elite, num_classes=action_dim).float().mean(dim=0)
        logits = CEM.cem_alpha * torch.log(freq + 1e-6) + (1.0 - CEM.cem_alpha) * logits
    return best_seq, best_score


@torch.no_grad()
def imagine(wm, h0, z0, seq, action_dim):
    h, z = h0, z0
    rews, obs, dones = [], [], []
    for t in range(len(seq)):
        a = F.one_hot(seq[t].view(1), num_classes=action_dim).float()
        h = wm.rssm.update_hidden(h, z, a)
        z, _ = wm.rssm.prior(h)
        obs.append(wm.reconstruct_obs(h, z)[0].cpu().numpy())
        rews.append(float(wm.predict_reward(h, z).item()))
        dones.append(float(torch.sigmoid(wm.predict_done_logits(h, z)).item())
                     if hasattr(wm, "predict_done_logits") else 0.0)
    return np.array(rews), np.array(obs), np.array(dones)


@torch.no_grad()
def warmup(wm, obs_ep, act_ep, t, action_dim):
    """Belief at obs[t] from W posterior steps on real observations (posterior mean)."""
    h = wm.rssm.init_hidden(1, DEVICE)
    z = torch.zeros(1, wm.rssm.latent_dim, device=DEVICE)
    h = wm.rssm.update_hidden(h, z, torch.zeros(1, action_dim, device=DEVICE))
    o = torch.tensor(obs_ep[t - W], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    z, _ = wm.rssm.posterior(h, o)
    for k in range(W):
        a = F.one_hot(torch.tensor([int(act_ep[t - W + k])], device=DEVICE), num_classes=action_dim).float()
        h = wm.rssm.update_hidden(h, z, a)
        o = torch.tensor(obs_ep[t - W + k + 1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z, _ = wm.rssm.posterior(h, o)
    return h, z


def build_start_states(val, n_eps, ts=(20, 100, 200)):
    ep_ids = np.unique(val["ep_index"])[:n_eps]
    starts = []
    for eid in ep_ids:
        i = np.where(val["ep_index"] == eid)[0]
        obs, act, rew = val["obs"][i], val["actions"][i], val["rewards"][i]
        contact = np.where((obs[:, 6] >= 0.5) | (obs[:, 7] >= 0.5))[0]
        fc = int(contact[0]) if len(contact) else len(obs)
        for t in ts:
            if t >= W and t + H < len(obs) and t + H < fc:
                starts.append(dict(ep=int(eid), seed=int(val["episode_seed"][i[0]]), t=t,
                                   obs=obs, act=act, rew=rew))
    return starts


def reconstruct_env(seed, act, t):
    env = gym.make("LunarLander-v3")
    o, _ = env.reset(seed=seed)
    for k in range(t):
        o, r, term, trunc, _ = env.step(int(act[k]))
        if term or trunc:
            raise RuntimeError("episode ended during reconstruction")
    return env, o


def real_rollout(env, seq):
    rews, obs, ended = [], [], False
    for a in seq:
        o, r, term, trunc, _ = env.step(int(a))
        rews.append(float(r)); obs.append(o)
        if term or trunc:
            ended = True; break
    rews = rews + [0.0] * (len(seq) - len(rews))
    return np.array(rews), np.array(obs), ended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_dataset", default="lunarlander_train_dataset.npz")
    ap.add_argument("--val_dataset", default="lunarlander_val_dataset.npz")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epoch_stride", type=int, default=25)
    ap.add_argument("--n_eps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--smoke", action="store_true", help="epochs 5 and 10, 3 episodes")
    args = ap.parse_args()

    train = np.load(args.train_dataset); val = np.load(args.val_dataset, allow_pickle=True)
    mu, sd = train["obs"].mean(0), train["obs"].std(0) + 1e-8
    ref = torch.tensor((train["obs"] - mu) / sd, dtype=torch.float32, device=DEVICE)

    @torch.no_grad()
    def supp(obs_arr):
        # mean over query points of the mean 5-NN Euclidean distance to the full
        # standardised training set (chunked brute force; no scipy on the cluster env)
        q = torch.tensor((np.asarray(obs_arr) - mu) / sd, dtype=torch.float32, device=DEVICE)
        d = torch.cdist(q, ref)                      # (nq, ntrain)
        knn = torch.topk(d, k=5, dim=1, largest=False).values
        return float(knn.mean().item())

    ckpts = []
    for f in os.listdir(args.checkpoints_dir):
        m = re.search(r"epoch_(\d+)\.pt$", f)
        if m and f.startswith("world_model_"):
            ckpts.append((int(m.group(1)), os.path.join(args.checkpoints_dir, f)))
    ckpts.sort()
    if args.smoke:
        ckpts = [c for c in ckpts if c[0] in (5, 10)]; n_eps = 3
    else:
        ckpts = [c for c in ckpts if c[0] % args.epoch_stride == 0]; n_eps = args.n_eps

    starts = build_start_states(val, n_eps)
    out = open(args.output, "w") if args.output else None
    hdr = ("epoch n_states gap_plan gap_human supp_plan supp_human supp_real_plan crash_plan "
           "score_plan done_plan r_img_plan r_real_plan r_img_human r_real_human")
    print(f"# run={args.run} starts={len(starts)} ckpts={len(ckpts)} config={args.config}")
    print(hdr)
    if out: out.write(f"# run={args.run} starts={len(starts)}\n{hdr}\n"); out.flush()

    obs_dim = val["obs"].shape[1]
    for epoch, path in ckpts:
        t0 = time.time()
        wm, action_dim = load_world_model(args.config, path, obs_dim)
        acc = {k: [] for k in hdr.split()[2:]}
        for i, s in enumerate(starts):
            set_seed((args.seed * 1000003 + epoch * 1009 + i) % (2 ** 32 - 1))
            h0, z0 = warmup(wm, s["obs"], s["act"], s["t"], action_dim)
            seq, score = cem_plan(wm, h0, z0, action_dim)
            r_img, o_img, d_img = imagine(wm, h0, z0, seq, action_dim)
            env, _ = reconstruct_env(s["seed"], s["act"], s["t"])
            r_real, o_real, ended = real_rollout(env, seq.cpu().numpy()); env.close()
            hseq = torch.tensor(s["act"][s["t"]:s["t"] + H], dtype=torch.long, device=DEVICE)
            r_img_h, o_img_h, _ = imagine(wm, h0, z0, hseq, action_dim)
            r_real_h = s["rew"][s["t"]:s["t"] + H]
            acc["gap_plan"].append(r_img.sum() - r_real.sum())
            acc["gap_human"].append(r_img_h.sum() - r_real_h.sum())
            acc["supp_plan"].append(supp(o_img)); acc["supp_human"].append(supp(o_img_h))
            acc["supp_real_plan"].append(supp(o_real) if len(o_real) else float("nan"))
            acc["crash_plan"].append(float(ended)); acc["score_plan"].append(score)
            acc["done_plan"].append(float(d_img.max()))
            acc["r_img_plan"].append(r_img.sum()); acc["r_real_plan"].append(r_real.sum())
            acc["r_img_human"].append(r_img_h.sum()); acc["r_real_human"].append(float(np.sum(r_real_h)))
        row = f"{epoch} {len(starts)} " + " ".join(f"{np.nanmean(acc[k]):.4f}" for k in hdr.split()[2:])
        print(row + f"   [{time.time() - t0:.0f}s]", flush=True)
        if out: out.write(row + "\n"); out.flush()
    if out: out.close()


if __name__ == "__main__":
    main()
