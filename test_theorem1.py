"""
Theorem-1 validity probes (designs D1, D2, D4 + leakage, fixed 2026-07-07).

Tests whether the trained nonlinear amortised RSSM behaves as its
linearisation promises at sampled belief states:

  D1 observation silence: perturb the state by eps along unobservable vs
     observable directions (matched eps), roll open-loop under identical
     ground-truth actions, measure predicted-observation divergence over the
     horizon; also the reward-head response along the same rollouts.
     Prediction, made before the run: median obs-response ratio
     (unobs/obs) < 0.15 at eps = 0.1, growing with eps.

  D2 filter corrects only observable errors: inject the same perturbations,
     then run the posterior (teacher-forced) chain with TRUE observations;
     measure error decay. Prediction, made before the run: observable-error
     decay >= 3x faster than unobservable-error decay. Side quantity: norm of the
     leakage block V_u^T A V_o relative to diagonal blocks.

  D4 consistency closure: over all right-singular directions, classify
     "measured silent" (nonlinear obs response below half the median
     observable response) vs analytic kernel membership; report agreement,
     and check reward-gradient mass in measured-silent directions ~ 1 - ROF.

All quantities are model-internal: no MPC-oracle contact.
"""

import argparse
import glob
import re

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import WorldModel, actions_to_vec
from train_models import SequenceDataset

DEVICE = "cpu"


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
    return wm, cfg


def collect_states(wm, loader, action_dim, n_states, warmup=5, horizon=25):
    """Warmed-up belief states plus the ground-truth action/obs tails after them."""
    rssm = wm.rssm
    out = []
    with torch.no_grad():
        for batch in loader:
            obs_seq, actions_seq, _, next_obs_seq, _, mask = [t.to(DEVICE) for t in batch]
            B, T = obs_seq.shape[:2]
            if T < warmup + horizon:
                continue
            h = rssm.init_hidden(B, DEVICE)
            z = torch.zeros(B, rssm.latent_dim, device=DEVICE)
            h = rssm.update_hidden(h, z, torch.zeros(B, action_dim, device=DEVICE))
            mu, _ = rssm.posterior(h, obs_seq[:, 0])
            z = mu
            for t in range(warmup - 1):
                a_t = actions_to_vec(actions_seq[:, t], action_dim)
                h = rssm.update_hidden(h, z, a_t)
                mu, _ = rssm.posterior(h, next_obs_seq[:, t])
                z = mu
            valid = mask[:, warmup + horizon - 1] > 0
            for i in valid.nonzero(as_tuple=True)[0][:4]:
                out.append((h[i:i + 1].clone(), z[i:i + 1].clone(),
                            actions_seq[i, warmup - 1:warmup - 1 + horizon].clone(),
                            next_obs_seq[i, warmup - 1:warmup - 1 + horizon].clone()))
            if len(out) >= n_states:
                return out[:n_states]
    return out


def local_bases(wm, h_i, z_i, action_dim, horizon):
    """Jacobians at the state; full SVD of O_H; observable/kernel bases + R."""
    rssm = wm.rssm
    hd = rssm.hidden_dim
    htop = rssm.top_hidden(h_i).squeeze(0)
    s = torch.cat([htop, z_i.squeeze(0)]).detach()

    def _trans(sf):
        hh, zz = sf[:hd].unsqueeze(0), sf[hd:].unsqueeze(0)
        hn = rssm.update_hidden(hh, zz, torch.zeros(1, action_dim, device=DEVICE))
        zn, _ = rssm.prior(hn)
        return torch.cat([rssm.top_hidden(hn).squeeze(0), zn.squeeze(0)])

    def _dec(sf):
        hh, zz = sf[:hd].unsqueeze(0), sf[hd:].unsqueeze(0)
        ph, cl, _ = wm.decode_heads(hh, zz)
        return torch.cat([ph.squeeze(0), torch.sigmoid(cl).squeeze(0)])

    def _rew(sf):
        hh, zz = sf[:hd].unsqueeze(0), sf[hd:].unsqueeze(0)
        return wm.predict_reward(hh, zz)

    A = torch.autograd.functional.jacobian(_trans, s)
    C = torch.autograd.functional.jacobian(_dec, s)
    R = torch.autograd.functional.jacobian(_rew, s).squeeze(0)
    rows, CAk = [C], C.clone()
    for _ in range(1, horizon):
        CAk = CAk @ A
        rows.append(CAk)
    Mo = torch.cat(rows, 0)
    _, S, Vh = torch.linalg.svd(Mo, full_matrices=True)
    k = int((S > S[0] * 1e-3).sum().item())
    return s, A, R, Vh, S, k


@torch.no_grad()
def rollout_obs_rew(wm, states, actions, horizon):
    """Batch open-loop rollout from full latent states [B, n]; returns stacked
    decoded observations [B, H, p] and rewards [B, H]."""
    rssm = wm.rssm
    hd = rssm.hidden_dim
    B = states.size(0)
    h = states[:, :hd]
    if rssm.gru_num_layers > 1:
        raise RuntimeError("probe assumes single-layer GRU")
    z = states[:, hd:]
    obs_out, rew_out = [], []
    for t in range(horizon):
        a_t = actions_to_vec(actions[t].unsqueeze(0), rssm.action_dim).expand(B, -1)
        h = rssm.update_hidden(h, z, a_t)
        z, _ = rssm.prior(h)
        ph, cl, _ = wm.decode_heads(h, z)
        obs_out.append(torch.cat([ph, torch.sigmoid(cl)], dim=-1))
        rew_out.append(wm.predict_reward(h, z).reshape(-1))
    return torch.stack(obs_out, 1), torch.stack(rew_out, 1)


@torch.no_grad()
def posterior_chain_error(wm, s0_true, s0_pert, actions, true_obs, steps):
    """Teacher-forced posterior updates on both chains; error norm over time."""
    rssm = wm.rssm
    hd = rssm.hidden_dim
    def step(h, z, a, o):
        h = rssm.update_hidden(h, z, a)
        mu, _ = rssm.posterior(h, o.unsqueeze(0))
        return h, mu
    h_t, z_t = s0_true[:, :hd], s0_true[:, hd:]
    h_p, z_p = s0_pert[:, :hd], s0_pert[:, hd:]
    errs = [float(torch.cat([h_p - h_t, z_p - z_t], -1).norm())]
    for t in range(steps):
        a_t = actions_to_vec(actions[t].unsqueeze(0), rssm.action_dim)
        h_t, z_t = step(h_t, z_t, a_t, true_obs[t])
        h_p, z_p = step(h_p, z_p, a_t, true_obs[t])
        errs.append(float(torch.cat([h_p - h_t, z_p - z_t], -1).norm()))
    return np.array(errs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--val_dataset", default="lunarlander_val_dataset.npz")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--n_states", type=int, default=12)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.3])
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))["world_model"]
    action_dim = int(cfg["action_dim"])
    ds = SequenceDataset(args.val_dataset, sequence_length=30, action_dim=action_dim,
                         random_start=False, dataset_seq_offset=7)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    for ck in args.checkpoints:
        wm, _ = load_wm(args.config, ck, ds.obs_dim)
        states = collect_states(wm, loader, action_dim, args.n_states,
                                horizon=args.horizon)
        d1_ratio = {e: [] for e in args.eps}
        d2_decay_obs, d2_decay_unobs, leak = [], [], []
        d4_agree, d4_mass_gap = [], []
        for (h_i, z_i, acts, tobs) in states:
            s, A, R, Vh, S, k = local_bases(wm, h_i, z_i, action_dim, args.horizon)
            n = s.numel()
            v_obs = Vh[[0, 1, 2, max(3, k // 2), max(4, k // 2 + 1), max(5, k - 1)], :]
            v_un = Vh[[n - 1, n - 2, n - 3, n - 4, n - 5, n - 6], :]
            base_obs, base_rew = rollout_obs_rew(wm, s.unsqueeze(0), acts, args.horizon)

            for e in args.eps:
                pert = torch.cat([s.unsqueeze(0) + e * v_obs,
                                  s.unsqueeze(0) + e * v_un], 0)
                o_p, r_p = rollout_obs_rew(wm, pert, acts, args.horizon)
                dobs = (o_p - base_obs).norm(dim=-1).mean(dim=-1)
                resp_o = dobs[:6].mean().item()
                resp_u = dobs[6:].mean().item()
                d1_ratio[e].append(resp_u / max(resp_o, 1e-9))

            e2 = 0.1
            s_obs = s.unsqueeze(0) + e2 * v_obs[0:1]
            s_un = s.unsqueeze(0) + e2 * v_un[0:1]
            err_o = posterior_chain_error(wm, s.unsqueeze(0), s_obs, acts, tobs, 12)
            err_u = posterior_chain_error(wm, s.unsqueeze(0), s_un, acts, tobs, 12)
            d2_decay_obs.append(err_o[10] / err_o[0])
            d2_decay_unobs.append(err_u[10] / err_u[0])

            B_ = torch.linalg.qr(Vh[:k, :].T)[0]
            U_ = torch.linalg.qr(Vh[k:, :].T)[0] if k < n else None
            if U_ is not None:
                A_oo = (B_.T @ A @ B_).norm()
                A_uo = (U_.T @ A @ B_).norm()
                leak.append(float(A_uo / max(A_oo, 1e-9)))

            e4 = 0.1
            pert_all = s.unsqueeze(0) + e4 * Vh
            o_all, _ = rollout_obs_rew(wm, pert_all, acts, args.horizon)
            resp_all = (o_all - base_obs).norm(dim=-1).mean(dim=-1)
            med_obs_resp = resp_all[:k].median()
            silent = resp_all < 0.5 * med_obs_resp
            analytic_kernel = torch.zeros(n, dtype=torch.bool)
            analytic_kernel[k:] = True
            d4_agree.append(float((silent == analytic_kernel).float().mean().item()))
            r_hat = (R / R.norm().clamp_min(1e-12))
            proj = (Vh @ r_hat).pow(2)
            mass_silent = float(proj[silent].sum().item())
            rof_analytic = float(proj[:k].sum().item())
            d4_mass_gap.append(abs(mass_silent - (1.0 - rof_analytic)))

        name = ck.split("/")[-1]
        print(f"\n== {name} (n_states={len(states)}) ==")
        for e in args.eps:
            r = np.array(d1_ratio[e])
            print(f"  D1 eps={e}: obs-response ratio unobs/obs median {np.median(r):.3f} "
                  f"(iqr {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f})")
        do, du = np.array(d2_decay_obs), np.array(d2_decay_unobs)
        print(f"  D2: error remaining at t=10 -- observable {np.median(do):.3f}, "
              f"unobservable {np.median(du):.3f} (decay-speed ratio {np.median(du/np.maximum(do,1e-9)):.1f}x)")
        print(f"  leakage |V_u^T A V_o| / |V_o^T A V_o|: median {np.median(leak):.3f}")
        print(f"  D4: silent-vs-kernel agreement median {np.median(d4_agree):.3f}; "
              f"|mass_silent - (1-ROF)| median {np.median(d4_mass_gap):.3f}")


if __name__ == "__main__":
    main()
