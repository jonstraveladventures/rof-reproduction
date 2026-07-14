"""
EXPLORATORY belief-drift diagnostics for world-model checkpoints.

Not part of the pre-registered CROF headline analysis. Motivated by the
active inference reading of ROF: open-loop imagination should stay on the
posterior's expected path (martingale property of calibrated beliefs), and
its systematic departure -- especially along value-relevant directions --
is a candidate offline predictor of closed-loop planning quality.

For each checkpoint, after the standard 5-step posterior warmup, two belief
chains are rolled under the SAME ground-truth actions:
  posterior chain: h/z updated with encoder corrections at every step
                   (deterministic: posterior means)
  open-loop chain: h/z rolled through the prior with no observations
                   (deterministic: prior means)

Per-step divergences, masked over valid positions, aggregated over a
25-step horizon (matching the paper's planning horizon):
  drift_kl:  KL( N(mu_post, s_post) || N(mu_prior, s_prior) )  [latent belief]
  drift_h:   ||h_post - h_ol|| / sqrt(dim)                     [hidden state]
  drift_rew: | r_theta(post belief) - r_theta(open-loop belief) |  [value-relevant]

Outputs per checkpoint: *_avg over horizon and *_end at the last step,
in the same "=== Epoch N ===" log format as eval_metrics.py.
"""

import argparse
import glob
import os
import re

import torch
import yaml
from torch.utils.data import DataLoader

from models import WorldModel, actions_to_vec
from train_models import SequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wm(config_path, ckpt, obs_dim):
    cfg = yaml.safe_load(open(config_path))["world_model"]
    cap = cfg["capacity"]
    wm = WorldModel(obs_dim=obs_dim, action_dim=int(cfg["action_dim"]),
                    latent_dim=cap["latent_dim"], hidden_dim=cap["hidden_dim"],
                    gru_num_layers=cap.get("gru_num_layers", 1),
                    mlp_hidden_dim=cap["mlp_hidden_dim"]).to(DEVICE)
    payload = torch.load(ckpt, map_location=DEVICE)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    wm.load_state_dict(sd)
    wm.eval()
    return wm


def gauss_kl(mu_q, logstd_q, mu_p, logstd_p):
    """KL(N_q || N_p), diagonal Gaussians, summed over latent dim."""
    var_q, var_p = (2 * logstd_q).exp(), (2 * logstd_p).exp()
    return 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p + 2 * (logstd_p - logstd_q) - 1.0).sum(-1)


@torch.no_grad()
def drift_for_checkpoint(wm, loader, warmup, horizon, max_seqs):
    rssm = wm.rssm
    latent_dim, action_dim = rssm.latent_dim, rssm.action_dim
    sums = {k: 0.0 for k in ("kl_avg", "kl_end", "h_avg", "h_end", "rew_avg", "rew_end")}
    n_used = 0
    for batch in loader:
        if n_used >= max_seqs:
            break
        obs_seq, actions_seq, _, next_obs_seq, _, mask = [t.to(DEVICE) for t in batch]
        B, T = obs_seq.shape[:2]
        if T < warmup + horizon:
            continue
        valid = mask[:, warmup + horizon - 1] > 0  # sequence survives the full horizon
        if not valid.any():
            continue

        h = rssm.init_hidden(B, DEVICE)
        z = torch.zeros(B, latent_dim, device=DEVICE)
        h = rssm.update_hidden(h, z, torch.zeros(B, action_dim, device=DEVICE))
        mu, _ = rssm.posterior(h, obs_seq[:, 0])
        z = mu
        for t in range(warmup - 1):
            a_t = actions_to_vec(actions_seq[:, t], action_dim)
            h = rssm.update_hidden(h, z, a_t)
            mu, _ = rssm.posterior(h, next_obs_seq[:, t])
            z = mu

        h_p, z_p = h, z          # posterior chain
        h_o, z_o = h.clone(), z.clone()  # open-loop chain
        kl_steps, h_steps, rew_steps = [], [], []
        for k in range(horizon):
            t = warmup - 1 + k
            a_t = actions_to_vec(actions_seq[:, t], action_dim)
            h_p = rssm.update_hidden(h_p, z_p, a_t)
            mu_post, logstd_post = rssm.posterior(h_p, next_obs_seq[:, t])
            h_o = rssm.update_hidden(h_o, z_o, a_t)
            mu_pr, logstd_pr = rssm.prior(h_o)
            kl = gauss_kl(mu_post, logstd_post, mu_pr, logstd_pr)
            htop_p, htop_o = rssm.top_hidden(h_p), rssm.top_hidden(h_o)
            hdist = (htop_p - htop_o).norm(dim=-1) / (htop_p.size(-1) ** 0.5)
            r_p = wm.predict_reward(h_p, mu_post).squeeze(-1)
            r_o = wm.predict_reward(h_o, mu_pr).squeeze(-1)
            rdiff = (r_p - r_o).abs()
            kl_steps.append(kl[valid].mean().item())
            h_steps.append(hdist[valid].mean().item())
            rew_steps.append(rdiff[valid].mean().item())
            z_p, z_o = mu_post, mu_pr

        nb = int(valid.sum().item())
        sums.setdefault("rew_scale", 0.0)
        sums["rew_scale"] += nb * r_p[valid].std().item()
        sums["kl_avg"] += nb * (sum(kl_steps) / horizon)
        sums["kl_end"] += nb * kl_steps[-1]
        sums["h_avg"] += nb * (sum(h_steps) / horizon)
        sums["h_end"] += nb * h_steps[-1]
        sums["rew_avg"] += nb * (sum(rew_steps) / horizon)
        sums["rew_end"] += nb * rew_steps[-1]
        n_used += nb
    out = {k: v / max(n_used, 1) for k, v in sums.items()}
    scale = max(out.pop("rew_scale", 1.0), 1e-6)
    out["rew_norm_avg"] = out["rew_avg"] / scale
    out["rew_norm_end"] = out["rew_end"] / scale
    return out, n_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_llc.yaml")
    p.add_argument("--val_dataset", default="lunarlander_continuous_val_dataset.npz")
    p.add_argument("--checkpoints_dir", required=True)
    p.add_argument("--epoch_min", type=int, default=None)
    p.add_argument("--epoch_max", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--max_seqs", type=int, default=2048)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))["world_model"]
    ds = SequenceDataset(args.val_dataset, sequence_length=30,
                         action_dim=int(cfg["action_dim"]), random_start=False,
                         dataset_seq_offset=5)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    obs_dim = ds.obs_dim

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

    with open(args.output, "w") as out:
        out.write(f"\nBelief-Drift Sweep (EXPLORATORY)\nCheckpoints: {len(paths)}, "
                  f"warmup={args.warmup}, horizon={args.horizon}, max_seqs={args.max_seqs}, "
                  f"seed={args.seed}\nDevice: {DEVICE}\n")
        for i, (e, f) in enumerate(paths, 1):
            wm = load_wm(args.config, f, obs_dim)
            vals, n = drift_for_checkpoint(wm, loader, args.warmup, args.horizon, args.max_seqs)
            line = (f"\n=== Epoch {e} === [{i}/{len(paths)}] (n={n})\n"
                    f"  drift_kl_avg={vals['kl_avg']:.4f} drift_kl_end={vals['kl_end']:.4f}\n"
                    f"  drift_h_avg={vals['h_avg']:.4f} drift_h_end={vals['h_end']:.4f}\n"
                    f"  drift_rew_avg={vals['rew_avg']:.4f} drift_rew_end={vals['rew_end']:.4f}\n"
                    f"  drift_rew_norm_avg={vals['rew_norm_avg']:.4f} drift_rew_norm_end={vals['rew_norm_end']:.4f}\n")
            out.write(line)
            out.flush()
            print(line, end="", flush=True)


if __name__ == "__main__":
    main()
