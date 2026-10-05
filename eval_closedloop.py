#!/usr/bin/env python3
"""Closed-loop probe: instrumented MPC episodes.

Design, predictions (written before the runs) and results:
docs/crof-heldout-and-planner-probe.md, section 3.
Reproduces the MPC-eval protocol (wm_mpc_policy.py at eval defaults: same seeds, same
belief update, same CEM, action 0 once landed, 600-step cap) and logs, per planner step,
the planner's predicted 25-step return of its chosen sequence, the realised 25-step return,
posterior reconstruction error on the current observation, and one-step model error along
the executed action. Aggregated per checkpoint by phase (near-ground: y < 0.35).

Usage:
    python eval_closedloop.py --run sp909 --checkpoints_dir /path/ckpt_sp909 --output results/cl_sp909.txt
"""
import argparse, os, re, random, time
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from eval_edge import load_world_model, cem_plan, imagine, set_seed, CEM, DEVICE, H

Y_NEAR = 0.35


def run_episode(wm, action_dim, env_seed, sd_obs, max_steps=600):
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=env_seed)
    set_seed(env_seed)
    h = wm.rssm.init_hidden(1, DEVICE)
    z = torch.zeros(1, wm.rssm.latent_dim, device=DEVICE)
    prev_action = 0
    rec = dict(rhat=[], recon=[], pred1=[], near=[], r=[], planner=[])
    ended, crash = False, False
    for step in range(1, max_steps + 1):
        landed = (obs[6] >= 0.5 and obs[7] >= 0.5 and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        pa = F.one_hot(torch.tensor([prev_action], device=DEVICE), num_classes=action_dim).float()
        with torch.no_grad():
            h, z, _, _, _, _ = wm.rssm.step(h, z, pa, obs_t)
            recon = float(((wm.reconstruct_obs(h, z)[0].cpu().numpy() - obs) / sd_obs).__pow__(2).mean() ** 0.5)
            if landed:
                action, rhat, o1 = 0, float("nan"), None
            else:
                seq, _ = cem_plan(wm, h, z, action_dim)
                r_img, o_img, _ = imagine(wm, h, z, seq, action_dim)
                action, rhat, o1 = int(seq[0].item()), float(r_img.sum()), o_img[0]
        next_obs, reward, term, trunc, _ = env.step(action)
        rec["r"].append(float(reward))
        rec["planner"].append(not landed)
        rec["rhat"].append(rhat)
        rec["recon"].append(recon)
        rec["pred1"].append(float((((o1 - next_obs) / sd_obs) ** 2).mean() ** 0.5) if o1 is not None else float("nan"))
        rec["near"].append(bool(obs[1] < Y_NEAR))
        if term or trunc:
            ended = True; crash = bool(term and reward <= -100.0); break
        obs, prev_action = next_obs, action
    env.close()
    r = np.array(rec["r"])
    realised = np.array([r[t:t + H].sum() for t in range(len(r))])
    return dict(ret=float(r.sum()), crash=crash, landed=(not crash) and ended,
                rhat=np.array(rec["rhat"]), realised=realised, recon=np.array(rec["recon"]),
                pred1=np.array(rec["pred1"]), near=np.array(rec["near"]), planner=np.array(rec["planner"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_dataset", default="lunarlander_train_dataset.npz")
    ap.add_argument("--output", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--epoch_stride", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--smoke", action="store_true", help="epoch 50 only, one episode")
    args = ap.parse_args()

    sd_obs = np.load(args.train_dataset)["obs"].std(0) + 1e-8
    ckpts = []
    for f in os.listdir(args.checkpoints_dir):
        m = re.search(r"epoch_(\d+)\.pt$", f)
        if m and f.startswith("world_model_"):
            ckpts.append((int(m.group(1)), os.path.join(args.checkpoints_dir, f)))
    ckpts.sort()
    if args.smoke:
        ckpts = [c for c in ckpts if c[0] == 50]; n_ep = 1
    else:
        ckpts = [c for c in ckpts if c[0] % args.epoch_stride == 0]; n_ep = args.episodes

    cols = ("epoch n_ep ep_return crash landed gap_flight gap_near rhat_flight rhat_near "
            "recon_flight recon_near pred1_flight pred1_near n_flight n_near")
    print(f"# run={args.run} ckpts={len(ckpts)} episodes={n_ep} config={args.config}")
    print(cols)
    out = open(args.output, "w") if args.output else None
    if out: out.write(f"# run={args.run} episodes={n_ep}\n{cols}\n"); out.flush()
    raw = {}
    for epoch, path in ckpts:
        t0 = time.time()
        wm, action_dim = load_world_model(args.config, path, 8)
        eps = [run_episode(wm, action_dim, args.seed + e, sd_obs) for e in range(1, n_ep + 1)]
        raw[epoch] = eps
        def pool(key, near_flag):
            vals = []
            for e in eps:
                m = e["planner"] & (e["near"] == near_flag)
                if key == "gap":
                    v = (e["rhat"] - e["realised"])[m]
                else:
                    v = e[key][m]
                vals.append(v[~np.isnan(v)])
            v = np.concatenate(vals) if vals else np.array([])
            return (float(v.mean()) if len(v) else float("nan")), len(v)
        gf, nf = pool("gap", False); gn, nn = pool("gap", True)
        row = (f"{epoch} {n_ep} {np.mean([e['ret'] for e in eps]):.2f} {np.mean([e['crash'] for e in eps]):.2f} "
               f"{np.mean([e['landed'] for e in eps]):.2f} {gf:.3f} {gn:.3f} {pool('rhat', False)[0]:.3f} {pool('rhat', True)[0]:.3f} "
               f"{pool('recon', False)[0]:.4f} {pool('recon', True)[0]:.4f} {pool('pred1', False)[0]:.4f} {pool('pred1', True)[0]:.4f} {nf} {nn}")
        print(row + f"   [{time.time() - t0:.0f}s]", flush=True)
        if out: out.write(row + "\n"); out.flush()
    if out:
        out.close()
        np.savez_compressed(args.output.replace(".txt", "_raw.npz"),
                            **{f"e{ep}_{i}_{k}": v for ep, eps in raw.items() for i, e in enumerate(eps)
                               for k, v in e.items() if isinstance(v, np.ndarray)})


if __name__ == "__main__":
    main()
