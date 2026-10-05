# ===========================================================================
# Reacher RSSM - Comprehensive metrics evaluation sweep over WM checkpoints.
#
# Port of LunarLander_RSSM/eval_metrics.py adapted for Reacher-v5:
#   - Continuous 2-D actions (no F.one_hot anywhere).
#   - 10-D obs decoded via WorldModel.decode_obs (cos/sin invariants
#     enforced by construction); no physics/contact split.
#   - No done head and no done loss (Reacher episodes are fixed-length).
#   - Reconstruction loss is per-dim mean MSE (matches train_models.py).
#   - 3-tuple loss_weights (recon, reward, kl); no done weight.
#   - Empirical controllability uses the four corner actions (+/-1, +/-1)
#     instead of one-hot identity.
#   - Empirical Lipschitz uses random state pairs (no per-action grouping).
#   - Default good/bad-return thresholds are calibrated to Reacher
#     (good >= -15 captures smooth_ik; bad <= -100 captures aggressive_ik
#     and iid_random; see dataset_audit_v2/report.txt).
#
# Metrics computed:
#   Loss decomposition:    val_loss, val_kl, val_recon, post_obs_rmse, post_rew_rmse
#   Open-loop rollout:     ol_obs_{start,avg,max,end}, ol_rew_{start,avg,max,end},
#                          ol_cumrew_err
#   Jacobian fixed:        jac_spec_radius[_max], jac_ctrl_rank/cond,
#                          jac_obs_rank/cond, jac_rcf, jac_ocf, jac_rof
#   Jacobian time-varying: jac_dyn_* counterparts (optional, --compute_dyn_jac)
#   ROF on bad episodes:   jac_rof_bad, jac_dyn_rof_bad
#   Empirical:             emp_C, emp_O, emp_L
#
# Output format mirrors the lander script (=== Epoch N === headers,
# space-separated key=value lines) so analyze_metrics.py can be ported
# unchanged.
#
# Usage:
#     python eval_metrics.py [--seed 12345]
# ===========================================================================

import argparse
import os
import re
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import WorldModel
from train_models import (
    SequenceDataset, kl_divergence, set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Validation loss decomposition (1-step posterior).
# =========================================================================
@torch.no_grad()
def compute_val_metrics(world_model, val_dataloader, beta_kl=0.5,
                        loss_weights=(1.0, 1.2, 1.0)):
    """1-step posterior validation pass.

    loss_weights = (recon, reward, kl). Reconstruction is per-dim mean MSE
    on the full 10-D obs (matches train_models.py). post_obs_rmse uses the
    same per-dim mean aggregation, so it is comparable to the rollout-MAE
    figures emitted during training.
    """
    was_training = world_model.training
    world_model.eval()

    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim
    device = next(world_model.parameters()).device

    total_kl = 0.0
    total_recon = 0.0
    total_rew_loss = 0.0
    total_mask = 0.0

    for batch in val_dataloader:
        obs_seq, actions_seq, rewards_seq, next_obs_seq, _dones_seq, mask = batch
        obs_seq = obs_seq.to(device)
        actions_seq = actions_seq.to(device)         # (B, T, action_dim) float
        rewards_seq = rewards_seq.to(device)
        next_obs_seq = next_obs_seq.to(device)
        mask = mask.to(device)

        batch_size, seq_len = obs_seq.shape[:2]
        h = rssm.init_hidden(batch_size, device)
        z = torch.zeros(batch_size, latent_dim, device=device)
        h = rssm.update_hidden(h, z, torch.zeros(batch_size, action_dim, device=device))
        mean_post, logstd_post = rssm.posterior(h, obs_seq[:, 0])
        z = rssm.sample_latent(mean_post, logstd_post)

        for t in range(seq_len):
            a_t = actions_seq[:, t]                  # continuous, no one-hot
            h = rssm.update_hidden(h, z, a_t)
            mean_prior, logstd_prior = rssm.prior(h)
            mean_post, logstd_post = rssm.posterior(h, next_obs_seq[:, t])
            z = rssm.sample_latent(mean_post, logstd_post)

            obs_pred = world_model.decode_obs(h, z)            # (B, 10)
            reward_pred = world_model.predict_reward(h, z)     # (B,)

            # per-dim mean MSE (matches training)
            recon_loss = (obs_pred - next_obs_seq[:, t]).pow(2).mean(-1)
            rew_loss = (reward_pred - rewards_seq[:, t]).pow(2)
            kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

            mask_t = mask[:, t]
            total_kl += (kl * mask_t).sum().item()
            total_recon += (recon_loss * mask_t).sum().item()
            total_rew_loss += (rew_loss * mask_t).sum().item()
            total_mask += mask_t.sum().item()

    world_model.train(was_training)

    if total_mask == 0:
        return {k: float('nan') for k in [
            'val_loss', 'val_kl', 'val_recon',
            'post_obs_rmse', 'post_rew_rmse']}

    weighted_loss = (
        loss_weights[0] * total_recon
        + loss_weights[1] * total_rew_loss
        + loss_weights[2] * beta_kl * total_kl
    ) / total_mask

    # post_obs_rmse = sqrt(val_recon): per-dim RMSE on the full 10-D obs.
    # Smaller-scale than lander's Euclidean RMSE by a factor of sqrt(obs_dim);
    # consistent within Reacher and matches the rollout-MAE figures from
    # train_models.py.
    return {
        'val_loss':       weighted_loss,
        'val_kl':         total_kl / total_mask,
        'val_recon':      total_recon / total_mask,
        'post_obs_rmse':  float(np.sqrt(total_recon    / total_mask)),
        'post_rew_rmse':  float(np.sqrt(total_rew_loss / total_mask)),
    }


# =========================================================================
# 2. Multi-step open-loop rollout RMSE.
# =========================================================================
@torch.no_grad()
def compute_multistep_rollout(world_model, val_dataloader,
                              warmup_steps=5, max_horizon=25):
    """Posterior warmup, then prior-only open-loop rollout with GT actions.

    Reports horizon-agnostic summary metrics:
      ol_obs_start  -- obs RMSE at step 1
      ol_obs_avg    -- obs RMSE averaged over all steps
      ol_obs_max    -- obs RMSE at worst step
      ol_obs_end    -- obs RMSE at final step (horizon H)
      ol_rew_start  -- reward RMSE at step 1
      ol_rew_avg    -- reward RMSE averaged over all steps
      ol_rew_max    -- reward RMSE at worst step
      ol_rew_end    -- reward RMSE at final step
      ol_cumrew_err -- cumulative reward RMSE over the rollout horizon
    """
    was_training = world_model.training
    world_model.eval()

    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim
    device = next(world_model.parameters()).device

    obs_errors = [[] for _ in range(max_horizon)]
    rew_errors = [[] for _ in range(max_horizon)]
    cumrew_pred_list = []
    cumrew_gt_list = []

    for batch in val_dataloader:
        obs_seq, actions_seq, rewards_seq, next_obs_seq, _dones_seq, mask = batch
        obs_seq = obs_seq.to(device)
        actions_seq = actions_seq.to(device)
        rewards_seq = rewards_seq.to(device)
        next_obs_seq = next_obs_seq.to(device)
        mask = mask.to(device)
        batch_size, seq_len = obs_seq.shape[:2]

        if seq_len < warmup_steps + max_horizon:
            continue

        h = rssm.init_hidden(batch_size, device)
        z = torch.zeros(batch_size, latent_dim, device=device)
        h = rssm.update_hidden(h, z, torch.zeros(batch_size, action_dim, device=device))
        mean_post, logstd_post = rssm.posterior(h, obs_seq[:, 0])
        z = rssm.sample_latent(mean_post, logstd_post)

        for t in range(warmup_steps):
            a_t = actions_seq[:, t]
            h = rssm.update_hidden(h, z, a_t)
            mean_post, logstd_post = rssm.posterior(h, next_obs_seq[:, t])
            z = rssm.sample_latent(mean_post, logstd_post)

        cumrew_pred = torch.zeros(batch_size, device=device)
        cumrew_gt = torch.zeros(batch_size, device=device)
        all_valid = torch.ones(batch_size, dtype=torch.bool, device=device)

        for k in range(max_horizon):
            t_idx = warmup_steps + k
            if t_idx >= seq_len:
                break

            a_k = actions_seq[:, t_idx]
            h = rssm.update_hidden(h, z, a_k)
            mean_prior, _ = rssm.prior(h)
            z = mean_prior              # deterministic open-loop

            obs_pred = world_model.decode_obs(h, z)
            rew_pred = world_model.predict_reward(h, z)

            valid = mask[:, t_idx] > 0
            all_valid &= valid
            if valid.any():
                # per-dim mean squared error -> averaged then sqrt'd later
                obs_err = (obs_pred[valid] - next_obs_seq[valid, t_idx]).pow(2).mean(-1)
                obs_errors[k].append(obs_err)
                rew_err = (rew_pred[valid] - rewards_seq[valid, t_idx]).pow(2)
                rew_errors[k].append(rew_err)

            valid_f = valid.float()
            cumrew_pred += rew_pred * valid_f
            cumrew_gt += rewards_seq[:, t_idx] * valid_f

        if all_valid.any():
            cumrew_pred_list.append(cumrew_pred[all_valid])
            cumrew_gt_list.append(cumrew_gt[all_valid])

    world_model.train(was_training)

    step_obs_rmse = []
    step_rew_rmse = []
    for k in range(max_horizon):
        if obs_errors[k]:
            step_obs_rmse.append(torch.cat(obs_errors[k]).mean().sqrt().item())
        else:
            step_obs_rmse.append(float('nan'))
        if rew_errors[k]:
            step_rew_rmse.append(torch.cat(rew_errors[k]).mean().sqrt().item())
        else:
            step_rew_rmse.append(float('nan'))

    valid_obs = [v for v in step_obs_rmse if not np.isnan(v)]
    valid_rew = [v for v in step_rew_rmse if not np.isnan(v)]

    result = {
        'ol_obs_start': step_obs_rmse[0]   if step_obs_rmse else float('nan'),
        'ol_obs_avg':   float(np.mean(valid_obs)) if valid_obs else float('nan'),
        'ol_obs_max':   float(np.max(valid_obs))  if valid_obs else float('nan'),
        'ol_obs_end':   step_obs_rmse[-1]  if step_obs_rmse else float('nan'),
        'ol_rew_start': step_rew_rmse[0]   if step_rew_rmse else float('nan'),
        'ol_rew_avg':   float(np.mean(valid_rew)) if valid_rew else float('nan'),
        'ol_rew_max':   float(np.max(valid_rew))  if valid_rew else float('nan'),
        'ol_rew_end':   step_rew_rmse[-1]  if step_rew_rmse else float('nan'),
    }

    if cumrew_pred_list:
        all_pred = torch.cat(cumrew_pred_list)
        all_gt = torch.cat(cumrew_gt_list)
        result['ol_cumrew_err'] = (all_pred - all_gt).pow(2).mean().sqrt().item()
    else:
        result['ol_cumrew_err'] = float('nan')

    return result


# =========================================================================
# 3. Jacobian-based controllability / observability analysis (fixed lin).
# =========================================================================
def hard_rof(Mat_o, r):
    """ROF by the frozen construction: fraction of |r|^2 in the span of the right
    singular vectors of Mat_o above 1e-3 of the largest singular value."""
    r_sq = r.pow(2).sum().item()
    _, S, Vh = torch.linalg.svd(Mat_o, full_matrices=False)
    k = int((S > S[0].item() * 1e-3).sum().item())
    if r_sq <= 1e-12 or k == 0:
        return float('nan')
    return (Vh[:k, :] @ r).pow(2).sum().item() / r_sq


def state_whiteners(world_model, val_dataset, n_states=1024, warmup_steps=5,
                    batch_size=64, seed=12345):
    """Square-root state covariances for the coordinate-invariant ROF variants.

    Port of the lander function, used here for the domain-contrast addendum to
    docs/crof-coordinate-invariance.md. Identical except
    that Reacher actions are continuous and fed to update_hidden directly.
    Returns {'wfull': Sigma^{1/2}, 'wdiag': diag(std)}. The caller saves and
    restores the global torch RNG around this call.
    """
    rssm = world_model.rssm
    action_dim, latent_dim = rssm.action_dim, rssm.latent_dim
    device = next(world_model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    states, n = [], 0
    with torch.no_grad():
        for batch in dl:
            if n >= n_states:
                break
            obs_seq, actions_seq, _, next_obs_seq, _, mask = batch
            obs_seq, actions_seq = obs_seq.to(device), actions_seq.to(device)
            next_obs_seq, mask = next_obs_seq.to(device), mask.to(device)
            bsz, seq_len = obs_seq.shape[:2]
            if seq_len < warmup_steps + 1:
                continue
            h = rssm.init_hidden(bsz, device)
            z = torch.zeros(bsz, latent_dim, device=device)
            h = rssm.update_hidden(h, z, torch.zeros(bsz, action_dim, device=device))
            mp, lp = rssm.posterior(h, obs_seq[:, 0])
            z = rssm.sample_latent(mp, lp)
            for t in range(warmup_steps):
                h = rssm.update_hidden(h, z, actions_seq[:, t])
                mp, lp = rssm.posterior(h, next_obs_seq[:, t])
                z = rssm.sample_latent(mp, lp)
            valid = (mask[:, warmup_steps] > 0).nonzero(as_tuple=True)[0][:n_states - n]
            states.append(torch.cat([rssm.top_hidden(h)[valid], z[valid]], dim=1))
            n += len(valid)
    X = torch.cat(states).double()
    X = X - X.mean(dim=0, keepdim=True)
    cov = X.T @ X / (X.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    w_full = (evecs * evals.clamp_min(0).sqrt()) @ evecs.T
    w_diag = torch.diag(cov.diagonal().clamp_min(0).sqrt())
    dtype = next(world_model.parameters()).dtype
    return {'wfull': w_full.to(dtype), 'wdiag': w_diag.to(dtype)}


def compute_jacobian_metrics(world_model, val_dataloader,
                             n_states=64, horizon=25, warmup_steps=5,
                             whiten=None):
    """Linearize the RSSM transition at sampled validation states.

    For each state, computes:
      A = d(h', z'_mean) / d(h, z)    transition Jacobian w.r.t. state
      B = d(h', z'_mean) / d(a)       transition Jacobian w.r.t. action  (B x action_dim)
      C = d(obs)         / d(h, z)    decoder Jacobian
      R = d(reward)      / d(h, z)    reward-head Jacobian

    Then derives spectral radius, controllability matrix rank/condition,
    observability matrix rank/condition, and the three subspace alignment
    fractions:
      RCF = || U_ctrl^T  R  || / || R ||           (reward in control subspace)
      OCF = || C  U_ctrl    || / || C ||           (obs in control subspace)
      ROF = || V_obs^T   R  || / || R ||           (reward in obs subspace)

    NOTE: state_flat is built from top_hidden(h) only -- assumes
    gru_num_layers == 1. For multi-layer GRUs the hidden state would
    need all layers; same caveat as the lander script.

    Reacher specifics: action_dim = 2, so the controllability matrix
    has 2*horizon columns (vs 4*horizon on lander), and ctrl_rank is
    upper-bounded by min(state_dim, 2*horizon).
    """
    was_training = world_model.training
    world_model.eval()

    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim
    hidden_dim = rssm.hidden_dim
    obs_dim = rssm.obs_dim
    state_dim = hidden_dim + latent_dim
    device = next(world_model.parameters()).device

    nan_result = {k: float('nan') for k in [
        'jac_spec_radius', 'jac_spec_radius_max',
        'jac_ctrl_rank', 'jac_ctrl_cond',
        'jac_obs_rank', 'jac_obs_cond',
        'jac_rcf', 'jac_ocf', 'jac_rof']}

    # --- Collect latent states via posterior warmup (1 sample per sequence) ---
    all_h, all_z, all_a = [], [], []
    n_collected = 0

    with torch.no_grad():
        for batch in val_dataloader:
            if n_collected >= n_states:
                break
            obs_seq, actions_seq, _, next_obs_seq, _, mask = batch
            obs_seq = obs_seq.to(device)
            actions_seq = actions_seq.to(device)
            next_obs_seq = next_obs_seq.to(device)
            mask = mask.to(device)
            batch_size, seq_len = obs_seq.shape[:2]

            if seq_len < warmup_steps + 1:
                continue

            h = rssm.init_hidden(batch_size, device)
            z = torch.zeros(batch_size, latent_dim, device=device)
            h = rssm.update_hidden(h, z, torch.zeros(batch_size, action_dim, device=device))
            mean_post, logstd_post = rssm.posterior(h, obs_seq[:, 0])
            z = rssm.sample_latent(mean_post, logstd_post)

            for t in range(warmup_steps):
                a_t = actions_seq[:, t]
                h = rssm.update_hidden(h, z, a_t)
                mean_post, logstd_post = rssm.posterior(h, next_obs_seq[:, t])
                z = rssm.sample_latent(mean_post, logstd_post)

            valid = mask[:, warmup_steps] > 0
            if valid.any() and n_collected < n_states:
                n_take = min(int(valid.sum().item()), n_states - n_collected)
                idx_v = valid.nonzero(as_tuple=True)[0][:n_take]
                all_h.append(h[idx_v].detach())
                all_z.append(z[idx_v].detach())
                all_a.append(actions_seq[idx_v, warmup_steps].detach())
                n_collected += n_take

    if n_collected == 0:
        world_model.train(was_training)
        return nan_result

    h_states = torch.cat(all_h)[:n_states]
    z_states = torch.cat(all_z)[:n_states]
    a_states = torch.cat(all_a)[:n_states]                # (N, action_dim) continuous
    N = h_states.size(0)

    # Disable parameter gradients for efficiency (we only need input Jacobians)
    param_grad_flags = {}
    for name, p in world_model.named_parameters():
        param_grad_flags[name] = p.requires_grad
        p.requires_grad_(False)

    spectral_radii = []
    ctrl_ranks = []
    ctrl_conds = []
    obs_ranks = []
    obs_conds = []
    rcf_list = []
    ocf_list = []
    rof_list = []
    rof_w_lists = {name: [] for name in (whiten or {})}

    try:
        for i in range(N):
            h_i = h_states[i:i+1].detach()
            z_i = z_states[i:i+1].detach()
            a_cont = a_states[i].detach()                 # (action_dim,) continuous

            h_top = rssm.top_hidden(h_i).squeeze(0).detach()
            z_flat = z_i.squeeze(0).detach()
            state_flat = torch.cat([h_top, z_flat])

            # A = d(next_state) / d(state)
            def _trans_state(sf):
                hh = sf[:hidden_dim].unsqueeze(0)
                zz = sf[hidden_dim:].unsqueeze(0)
                h_nxt = rssm.update_hidden(hh, zz, a_cont.unsqueeze(0))
                mp, _ = rssm.prior(h_nxt)
                return torch.cat([rssm.top_hidden(h_nxt).squeeze(0), mp.squeeze(0)])

            A_mat = torch.autograd.functional.jacobian(_trans_state, state_flat)

            # B = d(next_state) / d(action)
            def _trans_action(af):
                h_nxt = rssm.update_hidden(h_i, z_i, af.unsqueeze(0))
                mp, _ = rssm.prior(h_nxt)
                return torch.cat([rssm.top_hidden(h_nxt).squeeze(0), mp.squeeze(0)])

            B_mat = torch.autograd.functional.jacobian(_trans_action, a_cont)

            # C_obs = d(obs) / d(state)
            def _decode_state(sf):
                hh = sf[:hidden_dim].unsqueeze(0)
                zz = sf[hidden_dim:].unsqueeze(0)
                return world_model.decode_obs(hh, zz).squeeze(0)

            C_mat = torch.autograd.functional.jacobian(_decode_state, state_flat)

            # R = d(reward) / d(state)
            def _reward_state(sf):
                hh = sf[:hidden_dim].unsqueeze(0)
                zz = sf[hidden_dim:].unsqueeze(0)
                return world_model.predict_reward(hh, zz)

            R_vec = torch.autograd.functional.jacobian(
                _reward_state, state_flat).squeeze(0)

            # Spectral radius of A
            eigvals = torch.linalg.eigvals(A_mat)
            spectral_radii.append(eigvals.abs().max().item())

            # Controllability matrix [B, AB, A^2 B, ..., A^{H-1} B]
            cols = [B_mat]
            AkB = B_mat.clone()
            for _ in range(1, horizon):
                AkB = A_mat @ AkB
                cols.append(AkB)
            Mat_c = torch.cat(cols, dim=1)

            U_c, S_c, _ = torch.linalg.svd(Mat_c, full_matrices=False)
            thresh_c = S_c[0].item() * 1e-3
            k_ctrl = int((S_c > thresh_c).sum().item())
            ctrl_ranks.append(k_ctrl)
            n_sv_c = min(state_dim, action_dim * horizon)
            last_c = S_c[n_sv_c - 1].item() if n_sv_c <= len(S_c) else S_c[-1].item()
            ctrl_conds.append(S_c[0].item() / last_c if last_c > 1e-12 else float('inf'))

            # Observability matrix [C; CA; CA^2; ...; CA^{H-1}]
            rows = [C_mat]
            CAk = C_mat.clone()
            for _ in range(1, horizon):
                CAk = CAk @ A_mat
                rows.append(CAk)
            Mat_o = torch.cat(rows, dim=0)

            _, S_o, Vh_o = torch.linalg.svd(Mat_o, full_matrices=False)
            thresh_o = S_o[0].item() * 1e-3
            k_obs = int((S_o > thresh_o).sum().item())
            obs_ranks.append(k_obs)
            n_sv_o = min(obs_dim * horizon, state_dim)
            last_o = S_o[n_sv_o - 1].item() if n_sv_o <= len(S_o) else S_o[-1].item()
            obs_conds.append(S_o[0].item() / last_o if last_o > 1e-12 else float('inf'))

            # Controllable subspace basis (left singular vectors of M_c, top-k)
            U_ctrl = U_c[:, :k_ctrl] if k_ctrl > 0 else U_c[:, :1] * 0
            # Observable subspace basis (right singular vectors of M_o, top-k)
            V_obs = Vh_o[:k_obs, :].T if k_obs > 0 else Vh_o[:1, :].T * 0

            # RCF: Reward Controllability Fraction
            R_norm_sq = R_vec.pow(2).sum().item()
            if R_norm_sq > 1e-12 and k_ctrl > 0:
                rcf = (U_ctrl.T @ R_vec).pow(2).sum().item() / R_norm_sq
            else:
                rcf = float('nan')
            rcf_list.append(rcf)

            # OCF: Obs Controllability Fraction
            C_norm_sq = C_mat.pow(2).sum().item()
            if C_norm_sq > 1e-12 and k_ctrl > 0:
                ocf = (C_mat @ U_ctrl).pow(2).sum().item() / C_norm_sq
            else:
                ocf = float('nan')
            ocf_list.append(ocf)

            # ROF: Reward Observability Fraction
            if R_norm_sq > 1e-12 and k_obs > 0:
                rof = (V_obs.T @ R_vec).pow(2).sum().item() / R_norm_sq
            else:
                rof = float('nan')
            rof_list.append(rof)

            # Coordinate-invariant ROF (added 2026-09-24).
            for name, W in (whiten or {}).items():
                rof_w_lists[name].append(hard_rof(Mat_o @ W, W @ R_vec))

    finally:
        for name, p in world_model.named_parameters():
            p.requires_grad_(param_grad_flags[name])

    world_model.train(was_training)

    finite_conds_c = [c for c in ctrl_conds if np.isfinite(c)]
    finite_conds_o = [c for c in obs_conds if np.isfinite(c)]
    finite_rcf = [v for v in rcf_list if not np.isnan(v)]
    finite_ocf = [v for v in ocf_list if not np.isnan(v)]
    finite_rof = [v for v in rof_list if not np.isnan(v)]
    out_w = {}
    for name, vals in rof_w_lists.items():
        fin = [v for v in vals if not np.isnan(v)]
        out_w['jac_rof_' + name] = float(np.mean(fin)) if fin else float('nan')

    return {
        **out_w,
        'jac_spec_radius':     float(np.mean(spectral_radii)),
        'jac_spec_radius_max': float(np.max(spectral_radii)),
        'jac_ctrl_rank':       float(np.mean(ctrl_ranks)),
        'jac_ctrl_cond':       float(np.median(finite_conds_c)) if finite_conds_c else float('inf'),
        'jac_obs_rank':        float(np.mean(obs_ranks)),
        'jac_obs_cond':        float(np.median(finite_conds_o)) if finite_conds_o else float('inf'),
        'jac_rcf':             float(np.mean(finite_rcf)) if finite_rcf else float('nan'),
        'jac_ocf':             float(np.mean(finite_ocf)) if finite_ocf else float('nan'),
        'jac_rof':             float(np.mean(finite_rof)) if finite_rof else float('nan'),
    }


# =========================================================================
# 3b. Jacobian-based analysis with TIME-VARYING re-linearization.
# =========================================================================
def compute_jacobian_metrics_tv(world_model, val_dataloader,
                                n_states=64, horizon=25, warmup_steps=5):
    """Like compute_jacobian_metrics but re-linearizes at every rollout step.

    Instead of computing A once and using A^k, this rolls the latent state
    forward through the prior and computes fresh A_k, B_k, C_k, R_k at each
    step. The time-varying controllability / observability matrices are:

        Mat_c = [Phi(H,1) B_0 | Phi(H,2) B_1 | ... | B_{H-1}]
                where Phi(H,j+1) = A_{H-1} A_{H-2} ... A_{j+1}

        Mat_o = [C_0; C_1 A_0; C_2 A_1 A_0; ...; C_{H-1} A_{H-2}...A_0]

    Output keys use the jac_dyn_ prefix (jacobian-recomputed). On lander
    these were a weaker MPC predictor than the fixed variant, hence
    --compute_dyn_jac defaults to False; same caveat applies for Reacher.
    """
    was_training = world_model.training
    world_model.eval()

    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim
    hidden_dim = rssm.hidden_dim
    obs_dim = rssm.obs_dim
    state_dim = hidden_dim + latent_dim
    device = next(world_model.parameters()).device

    nan_keys = [
        'jac_dyn_spec_radius', 'jac_dyn_spec_radius_max',
        'jac_dyn_ctrl_rank', 'jac_dyn_ctrl_cond',
        'jac_dyn_obs_rank', 'jac_dyn_obs_cond',
        'jac_dyn_rcf', 'jac_dyn_ocf', 'jac_dyn_rof']
    nan_result = {k: float('nan') for k in nan_keys}

    # --- Collect starting states AND subsequent actions for rollout ---
    all_h, all_z, all_actions_seq = [], [], []
    n_collected = 0

    with torch.no_grad():
        for batch in val_dataloader:
            if n_collected >= n_states:
                break
            obs_seq, actions_seq, _, next_obs_seq, _, mask = batch
            obs_seq = obs_seq.to(device)
            actions_seq = actions_seq.to(device)
            next_obs_seq = next_obs_seq.to(device)
            mask = mask.to(device)
            batch_size, seq_len = obs_seq.shape[:2]

            if seq_len < warmup_steps + horizon:
                continue

            h = rssm.init_hidden(batch_size, device)
            z = torch.zeros(batch_size, latent_dim, device=device)
            h = rssm.update_hidden(h, z,
                    torch.zeros(batch_size, action_dim, device=device))
            mean_post, logstd_post = rssm.posterior(h, obs_seq[:, 0])
            z = rssm.sample_latent(mean_post, logstd_post)

            for t in range(warmup_steps):
                a_t = actions_seq[:, t]
                h = rssm.update_hidden(h, z, a_t)
                mean_post, logstd_post = rssm.posterior(h, next_obs_seq[:, t])
                z = rssm.sample_latent(mean_post, logstd_post)

            valid = mask[:, warmup_steps] > 0
            if valid.any() and n_collected < n_states:
                n_take = min(int(valid.sum().item()), n_states - n_collected)
                idx_v = valid.nonzero(as_tuple=True)[0][:n_take]
                all_h.append(h[idx_v].detach())
                all_z.append(z[idx_v].detach())
                future_acts = actions_seq[idx_v, warmup_steps:
                                          warmup_steps + horizon].detach()
                all_actions_seq.append(future_acts)
                n_collected += n_take

    if n_collected == 0:
        world_model.train(was_training)
        return nan_result

    h_states = torch.cat(all_h)[:n_states]
    z_states = torch.cat(all_z)[:n_states]
    act_seqs = torch.cat(all_actions_seq)[:n_states]      # (N, H, action_dim)
    N = h_states.size(0)

    param_grad_flags = {}
    for name, p in world_model.named_parameters():
        param_grad_flags[name] = p.requires_grad
        p.requires_grad_(False)

    spectral_radii = []
    ctrl_ranks, ctrl_conds = [], []
    obs_ranks, obs_conds = [], []
    rcf_list, ocf_list, rof_list = [], [], []

    try:
        for i in range(N):
            h_cur = h_states[i:i+1].detach()
            z_cur = z_states[i:i+1].detach()

            A_list, B_list, C_list, R_list = [], [], [], []
            spec_radii_i = []

            for k in range(horizon):
                a_cont = act_seqs[i, k].detach()          # (action_dim,) continuous

                h_top = rssm.top_hidden(h_cur).squeeze(0).detach()
                z_flat = z_cur.squeeze(0).detach()
                state_flat = torch.cat([h_top, z_flat])

                def _trans_state(sf, _a=a_cont):
                    hh = sf[:hidden_dim].unsqueeze(0)
                    zz = sf[hidden_dim:].unsqueeze(0)
                    h_nxt = rssm.update_hidden(hh, zz, _a.unsqueeze(0))
                    mp, _ = rssm.prior(h_nxt)
                    return torch.cat([rssm.top_hidden(h_nxt).squeeze(0),
                                      mp.squeeze(0)])

                A_k = torch.autograd.functional.jacobian(
                    _trans_state, state_flat)

                def _trans_action(af, _h=h_cur, _z=z_cur):
                    h_nxt = rssm.update_hidden(_h, _z, af.unsqueeze(0))
                    mp, _ = rssm.prior(h_nxt)
                    return torch.cat([rssm.top_hidden(h_nxt).squeeze(0),
                                      mp.squeeze(0)])

                B_k = torch.autograd.functional.jacobian(
                    _trans_action, a_cont)

                def _decode_state(sf):
                    hh = sf[:hidden_dim].unsqueeze(0)
                    zz = sf[hidden_dim:].unsqueeze(0)
                    return world_model.decode_obs(hh, zz).squeeze(0)

                C_k = torch.autograd.functional.jacobian(
                    _decode_state, state_flat)

                def _reward_state(sf):
                    hh = sf[:hidden_dim].unsqueeze(0)
                    zz = sf[hidden_dim:].unsqueeze(0)
                    return world_model.predict_reward(hh, zz)

                R_k = torch.autograd.functional.jacobian(
                    _reward_state, state_flat).squeeze(0)

                A_list.append(A_k.detach())
                B_list.append(B_k.detach())
                C_list.append(C_k.detach())
                R_list.append(R_k.detach())

                eigvals = torch.linalg.eigvals(A_k)
                spec_radii_i.append(eigvals.abs().max().item())

                # Step forward using the prior mean (deterministic)
                with torch.no_grad():
                    h_cur = rssm.update_hidden(h_cur, z_cur,
                                               a_cont.unsqueeze(0))
                    z_cur, _ = rssm.prior(h_cur)

            spectral_radii.extend(spec_radii_i)

            # --- Time-varying controllability matrix ---
            # Column block j: Phi(H, j+1) * B_j
            # where Phi(H, j+1) = A_{H-1} * A_{H-2} * ... * A_{j+1}
            ctrl_cols = []
            for j in range(horizon):
                col = B_list[j].clone()
                for t in range(j + 1, horizon):
                    col = A_list[t] @ col
                ctrl_cols.append(col)
            Mat_c = torch.cat(ctrl_cols, dim=1)

            U_c, S_c, _ = torch.linalg.svd(Mat_c, full_matrices=False)
            thresh_c = S_c[0].item() * 1e-3
            k_ctrl = int((S_c > thresh_c).sum().item())
            ctrl_ranks.append(k_ctrl)
            n_sv_c = min(state_dim, action_dim * horizon)
            last_c = (S_c[n_sv_c - 1].item() if n_sv_c <= len(S_c)
                       else S_c[-1].item())
            ctrl_conds.append(S_c[0].item() / last_c
                              if last_c > 1e-12 else float('inf'))

            # --- Time-varying observability matrix ---
            # Row block k: C_k * Phi(k, 0)
            # where Phi(k, 0) = A_{k-1} * ... * A_0, Phi(0,0) = I
            obs_rows = [C_list[0]]
            Phi = torch.eye(state_dim, device=device)
            for k in range(1, horizon):
                Phi = A_list[k - 1] @ Phi
                obs_rows.append(C_list[k] @ Phi)
            Mat_o = torch.cat(obs_rows, dim=0)

            _, S_o, Vh_o = torch.linalg.svd(Mat_o, full_matrices=False)
            thresh_o = S_o[0].item() * 1e-3
            k_obs = int((S_o > thresh_o).sum().item())
            obs_ranks.append(k_obs)
            n_sv_o = min(obs_dim * horizon, state_dim)
            last_o = (S_o[n_sv_o - 1].item() if n_sv_o <= len(S_o)
                       else S_o[-1].item())
            obs_conds.append(S_o[0].item() / last_o
                             if last_o > 1e-12 else float('inf'))

            # Subspace bases
            U_ctrl = U_c[:, :k_ctrl] if k_ctrl > 0 else U_c[:, :1] * 0
            V_obs = Vh_o[:k_obs, :].T if k_obs > 0 else Vh_o[:1, :].T * 0

            # Use R from the initial state for RCF / ROF (matches lander)
            R_vec = R_list[0]
            R_norm_sq = R_vec.pow(2).sum().item()

            if R_norm_sq > 1e-12 and k_ctrl > 0:
                rcf = (U_ctrl.T @ R_vec).pow(2).sum().item() / R_norm_sq
            else:
                rcf = float('nan')
            rcf_list.append(rcf)

            C_0 = C_list[0]
            C_norm_sq = C_0.pow(2).sum().item()
            if C_norm_sq > 1e-12 and k_ctrl > 0:
                ocf = (C_0 @ U_ctrl).pow(2).sum().item() / C_norm_sq
            else:
                ocf = float('nan')
            ocf_list.append(ocf)

            if R_norm_sq > 1e-12 and k_obs > 0:
                rof = (V_obs.T @ R_vec).pow(2).sum().item() / R_norm_sq
            else:
                rof = float('nan')
            rof_list.append(rof)

    finally:
        for name, p in world_model.named_parameters():
            p.requires_grad_(param_grad_flags[name])

    world_model.train(was_training)

    finite_conds_c = [c for c in ctrl_conds if np.isfinite(c)]
    finite_conds_o = [c for c in obs_conds if np.isfinite(c)]
    finite_rcf = [v for v in rcf_list if not np.isnan(v)]
    finite_ocf = [v for v in ocf_list if not np.isnan(v)]
    finite_rof = [v for v in rof_list if not np.isnan(v)]

    return {
        'jac_dyn_spec_radius':     float(np.mean(spectral_radii)),
        'jac_dyn_spec_radius_max': float(np.max(spectral_radii)),
        'jac_dyn_ctrl_rank':       float(np.mean(ctrl_ranks)),
        'jac_dyn_ctrl_cond':       float(np.median(finite_conds_c)) if finite_conds_c else float('inf'),
        'jac_dyn_obs_rank':        float(np.mean(obs_ranks)),
        'jac_dyn_obs_cond':        float(np.median(finite_conds_o)) if finite_conds_o else float('inf'),
        'jac_dyn_rcf':             float(np.mean(finite_rcf)) if finite_rcf else float('nan'),
        'jac_dyn_ocf':             float(np.mean(finite_ocf)) if finite_ocf else float('nan'),
        'jac_dyn_rof':             float(np.mean(finite_rof)) if finite_rof else float('nan'),
    }


# =========================================================================
# 4. Empirical sensitivity metrics (C, O, L) for continuous actions.
# =========================================================================
@torch.no_grad()
def compute_control_metrics(world_model, val_dataloader,
                            max_states=256, n_obs_samples=64, epsilon=1e-3,
                            max_lip_pairs=1000, seed=12345):
    """Empirical control-theoretic adequacy metrics.

    Lander used categorical actions, so its Controllability metric
    enumerated all action one-hots and the Lipschitz metric grouped
    pairs by discrete action id. Reacher actions are continuous in
    [-1, 1]^2 so we adapt as follows:

      1. Controllability C -- pairwise next-state distance across the
         four CORNER actions { (-1,-1), (-1,+1), (+1,-1), (+1,+1) }.
         This discretizes the action box into the four extreme actions
         CEM tends to converge to (saturated boundary), giving a clean
         continuous analogue of the lander metric. 6 ordered pairs;
         per-state score is their mean L2 distance.

      2. Observability O -- perturb z by epsilon along each latent dim,
         measure mean state-space sensitivity. Identical to lander.

      3. Lipschitz L -- random state pairs (no per-action grouping;
         the categorical-action equivalence class doesn't exist for
         continuous actions). Apply each pair's first-state action to
         BOTH states; record the (output_distance / input_distance)
         ratio. Use lip_max as the headline Lipschitz number, matching
         lander's choice (max captures localized blowups that mean
         smooths over).
    """
    was_training = world_model.training
    world_model.eval()
    device = next(world_model.parameters()).device
    rssm = world_model.rssm
    action_dim = rssm.action_dim
    latent_dim = rssm.latent_dim

    def _prior_step(h, z, a_cont):
        h_next = rssm.update_hidden(h, z, a_cont)
        mean_prior, _ = rssm.prior(h_next)
        return h_next, mean_prior

    def _state_cat(h, z):
        return torch.cat([rssm.top_hidden(h), z], dim=-1)

    collected_h, collected_z, collected_a = [], [], []
    n_collected = 0

    for batch in val_dataloader:
        obs_seq, actions_seq, _, next_obs_seq, _, mask = batch
        obs_seq = obs_seq.to(device)
        actions_seq = actions_seq.to(device)
        next_obs_seq = next_obs_seq.to(device)
        mask = mask.to(device)
        batch_size, seq_len = obs_seq.shape[:2]

        h = rssm.init_hidden(batch_size, device)
        z = torch.zeros(batch_size, latent_dim, device=device)
        h = rssm.update_hidden(h, z, torch.zeros(batch_size, action_dim, device=device))
        mean_post, logstd_post = rssm.posterior(h, obs_seq[:, 0])
        z = rssm.sample_latent(mean_post, logstd_post)

        for t in range(seq_len):
            a_t = actions_seq[:, t]
            h = rssm.update_hidden(h, z, a_t)
            mean_post, logstd_post = rssm.posterior(h, next_obs_seq[:, t])
            z = rssm.sample_latent(mean_post, logstd_post)

            valid = mask[:, t] > 0
            if valid.any() and n_collected < max_states:
                n_take = min(int(valid.sum().item()), max_states - n_collected)
                idx_valid = valid.nonzero(as_tuple=True)[0][:n_take]
                collected_h.append(h[idx_valid])
                collected_z.append(z[idx_valid])
                collected_a.append(actions_seq[idx_valid, t])
                n_collected += n_take

    nan_result = {k: float('nan') for k in [
        'ctrl_mean', 'ctrl_min', 'obs_mean', 'obs_max',
        'lip_max', 'lip_mean']}
    if not collected_h:
        world_model.train(was_training)
        return nan_result

    all_h = torch.cat(collected_h)[:max_states]
    all_z = torch.cat(collected_z)[:max_states]
    all_a = torch.cat(collected_a)[:max_states]            # (N, action_dim) continuous
    N = all_h.size(0)

    # Metric 1: Controllability across the 4 corner actions of [-1, 1]^2.
    # Hard-coded to action_dim == 2 (Reacher); if Reacher's action_dim ever
    # changes we'd need to revisit the corner enumeration (2^action_dim
    # corners -> action_dim*(action_dim-1)/2 ... no, 2^d corners and
    # 2^d * (2^d - 1) / 2 pairs; grows fast for higher d).
    if action_dim != 2:
        # Fallback: random N_RAND actions if action_dim is something other
        # than 2. Keeps the metric meaningful if Reacher gets generalized.
        n_corners = 8
        rng = torch.Generator(device=device).manual_seed(seed)
        corners = torch.empty(n_corners, action_dim, device=device).uniform_(
            -1.0, 1.0, generator=rng)
    else:
        corners = torch.tensor([
            [-1.0, -1.0],
            [-1.0,  1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
        ], device=device, dtype=all_a.dtype)
        n_corners = 4

    h_rep = all_h.repeat_interleave(n_corners, dim=0)
    z_rep = all_z.repeat_interleave(n_corners, dim=0)
    a_all = corners.unsqueeze(0).expand(N, -1, -1).reshape(N * n_corners, -1)
    h_next_all, z_next_all = _prior_step(h_rep, z_rep, a_all)
    next_states = _state_cat(h_next_all, z_next_all).reshape(N, n_corners, -1)

    ctrl_per_state = torch.zeros(N, device=device)
    n_pairs_corner = n_corners * (n_corners - 1) // 2
    for i in range(n_corners):
        for j in range(i + 1, n_corners):
            ctrl_per_state += (next_states[:, i] - next_states[:, j]).norm(dim=-1)
    ctrl_per_state /= max(n_pairs_corner, 1)
    ctrl_mean = ctrl_per_state.mean().item()
    ctrl_min = ctrl_per_state.min().item()

    # Metric 2: Observability (perturb z only by epsilon along each latent dim)
    rng = torch.Generator(device=device).manual_seed(seed)
    n_obs = min(N, n_obs_samples)
    obs_idx = torch.randperm(N, device=device, generator=rng)[:n_obs]
    obs_scores = torch.zeros(n_obs, device=device)

    for pos in range(n_obs):
        si = obs_idx[pos]
        h_i = all_h[si:si + 1]
        z_i = all_z[si:si + 1]
        a_i = all_a[si:si + 1]                              # (1, action_dim)

        h_base, z_base = _prior_step(h_i, z_i, a_i)
        s_base = _state_cat(h_base, z_base)

        z_pert = z_i.expand(latent_dim, -1).clone()
        z_pert += epsilon * torch.eye(latent_dim, device=device)
        h_for_z = h_i.repeat(latent_dim, *[1] * (h_i.dim() - 1))
        a_for_z = a_i.expand(latent_dim, -1)
        h_out_z, z_out_z = _prior_step(h_for_z, z_pert, a_for_z)
        diffs_z = (_state_cat(h_out_z, z_out_z) - s_base).norm(dim=-1) / epsilon

        obs_scores[pos] = diffs_z.mean()

    obs_mean = obs_scores.mean().item()
    obs_max = obs_scores.max().item()

    # Metric 3: Lipschitz on RANDOM state pairs (continuous actions: no
    # action-id grouping). For each pair (s_a, s_b), apply s_a's stored
    # action to BOTH states and measure how the model dilates the
    # input-state distance.
    eps_L = 1e-6
    if N >= 2:
        n_pairs = min(N * (N - 1) // 2, max_lip_pairs)
        idx_a = torch.randint(0, N, (n_pairs,), device=device, generator=rng)
        idx_b = torch.randint(0, N, (n_pairs,), device=device, generator=rng)
        valid_pairs = idx_a != idx_b
        idx_a, idx_b = idx_a[valid_pairs], idx_b[valid_pairs]

        if idx_a.numel() > 0:
            s_a = _state_cat(all_h[idx_a], all_z[idx_a])
            s_b = _state_cat(all_h[idx_b], all_z[idx_b])
            d_in = (s_a - s_b).norm(dim=-1)

            a_pair = all_a[idx_a]                          # (n_pairs, action_dim)
            h_na, z_na = _prior_step(all_h[idx_a], all_z[idx_a], a_pair)
            h_nb, z_nb = _prior_step(all_h[idx_b], all_z[idx_b], a_pair)
            d_out = (_state_cat(h_na, z_na) - _state_cat(h_nb, z_nb)).norm(dim=-1)

            ratios = d_out / (d_in + eps_L)
            lip_max = ratios.max().item()
            lip_mean = ratios.mean().item()
        else:
            lip_max = lip_mean = float('nan')
    else:
        lip_max = lip_mean = float('nan')

    world_model.train(was_training)
    return {
        'ctrl_mean': ctrl_mean,
        'ctrl_min':  ctrl_min,
        'obs_mean':  obs_mean,
        'obs_max':   obs_max,
        'lip_max':   lip_max,
        'lip_mean':  lip_mean,
    }


# =========================================================================
# Formatting helpers.
# =========================================================================
def format_metrics(metrics):
    """Format all metrics into parseable log lines (matches lander layout)."""
    lines = []

    # Loss decomposition + posterior (closed-loop, 1-step) prediction errors
    lines.append(
        "val_loss={val_loss:.4f} val_kl={val_kl:.4f} val_recon={val_recon:.4f}"
        " post_obs_rmse={post_obs_rmse:.4f} post_rew_rmse={post_rew_rmse:.4f}"
        .format(**metrics))

    # Open-loop (prior-only) multi-step rollout observation RMSE
    lines.append(
        "ol_obs_start={:.4f} ol_obs_avg={:.4f} ol_obs_max={:.4f} ol_obs_end={:.4f}".format(
            metrics.get('ol_obs_start', float('nan')),
            metrics.get('ol_obs_avg', float('nan')),
            metrics.get('ol_obs_max', float('nan')),
            metrics.get('ol_obs_end', float('nan'))))

    # Open-loop multi-step rollout reward RMSE
    lines.append(
        "ol_rew_start={:.4f} ol_rew_avg={:.4f} ol_rew_max={:.4f} ol_rew_end={:.4f}".format(
            metrics.get('ol_rew_start', float('nan')),
            metrics.get('ol_rew_avg', float('nan')),
            metrics.get('ol_rew_max', float('nan')),
            metrics.get('ol_rew_end', float('nan'))))

    lines.append("ol_cumrew_err={:.4f}".format(
        metrics.get('ol_cumrew_err', float('nan'))))

    # Jacobian-based: structure
    lines.append(
        "jac_spec_radius={:.4f} jac_spec_radius_max={:.4f}"
        " jac_ctrl_rank={:.1f} jac_ctrl_cond={:.1f}"
        " jac_obs_rank={:.1f} jac_obs_cond={:.1f}".format(
            metrics.get('jac_spec_radius', float('nan')),
            metrics.get('jac_spec_radius_max', float('nan')),
            metrics.get('jac_ctrl_rank', float('nan')),
            metrics.get('jac_ctrl_cond', float('nan')),
            metrics.get('jac_obs_rank', float('nan')),
            metrics.get('jac_obs_cond', float('nan'))))

    # Jacobian-based: subspace alignment
    lines.append(
        "jac_rcf={:.4f} jac_ocf={:.4f} jac_rof={:.4f}".format(
            metrics.get('jac_rcf', float('nan')),
            metrics.get('jac_ocf', float('nan')),
            metrics.get('jac_rof', float('nan'))))

    # Jacobian dynamic (time-varying): structure
    if 'jac_dyn_spec_radius' in metrics:
        lines.append(
            "jac_dyn_spec_radius={:.4f} jac_dyn_spec_radius_max={:.4f}"
            " jac_dyn_ctrl_rank={:.1f} jac_dyn_ctrl_cond={:.1f}"
            " jac_dyn_obs_rank={:.1f} jac_dyn_obs_cond={:.1f}".format(
                metrics.get('jac_dyn_spec_radius', float('nan')),
                metrics.get('jac_dyn_spec_radius_max', float('nan')),
                metrics.get('jac_dyn_ctrl_rank', float('nan')),
                metrics.get('jac_dyn_ctrl_cond', float('nan')),
                metrics.get('jac_dyn_obs_rank', float('nan')),
                metrics.get('jac_dyn_obs_cond', float('nan'))))
        lines.append(
            "jac_dyn_rcf={:.4f} jac_dyn_ocf={:.4f} jac_dyn_rof={:.4f}".format(
                metrics.get('jac_dyn_rcf', float('nan')),
                metrics.get('jac_dyn_ocf', float('nan')),
                metrics.get('jac_dyn_rof', float('nan'))))

    # Jacobian ROF on bad episodes (curated sampling)
    if 'jac_rof_bad' in metrics:
        lines.append(
            "jac_rof_bad={:.4f} jac_dyn_rof_bad={:.4f}".format(
                metrics.get('jac_rof_bad', float('nan')),
                metrics.get('jac_dyn_rof_bad', float('nan'))))

    if 'jac_rof_wfull' in metrics:
        lines.append(
            "jac_rof_wfull={:.4f} jac_rof_wdiag={:.4f}"
            " jac_rof_wfull_bad={:.4f} jac_rof_wdiag_bad={:.4f}".format(
                metrics.get('jac_rof_wfull', float('nan')),
                metrics.get('jac_rof_wdiag', float('nan')),
                metrics.get('jac_rof_wfull_bad', float('nan')),
                metrics.get('jac_rof_wdiag_bad', float('nan'))))

    # Empirical
    lines.append(
        "emp_C={:.4f} emp_O={:.4f} emp_L={:.4f}".format(
            metrics.get('emp_C', float('nan')),
            metrics.get('emp_O', float('nan')),
            metrics.get('emp_L', float('nan'))))

    return "\n  ".join(lines)


# =========================================================================
# Main.
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive metrics for every Reacher world-model checkpoint")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--val_dataset", default="reacher_val_dataset.npz")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--output", default="metrics_eval_logs.txt")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epoch_min", type=int, default=None)
    parser.add_argument("--epoch_max", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--epoch_stride", type=int, default=1,
                        help="Take every N-th checkpoint (1=all)")
    parser.add_argument("--n_jac_states", type=int, default=64,
                        help="Number of latent states for Jacobian analysis")
    # Reacher returns are negative (reward = -dist - 0.01||a||^2). The lander
    # used a single symmetric threshold (`--good_ep_return X` -> good >= X,
    # bad <= -X) which doesn't fit Reacher; use two explicit thresholds.
    # Defaults are calibrated to dataset_audit_v2/report.txt:
    #   smooth_ik returns: -4.3 to -13.9   -> all qualify as 'good' at -15
    #   iid_random:        -78 to -98      -> qualify as 'bad' at -100
    #   aggressive_ik:     -123 to -214    -> qualify as 'bad' at -100
    parser.add_argument("--good_min_return", type=float, default=-15.0,
                        help="Min episode return to qualify as 'good' for "
                             "curated Jacobian sampling (Reacher: -15 captures smooth_ik)")
    parser.add_argument("--bad_max_return", type=float, default=-100.0,
                        help="Max episode return to qualify as 'bad' for "
                             "curated Jacobian sampling (Reacher: -100 captures "
                             "iid_random + aggressive_ik buckets)")
    parser.add_argument("--whiten", action="store_true",
                        help="also compute the coordinate-invariant ROF variants "
                             "(docs/crof-coordinate-invariance.md); leaves every other value unchanged")
    parser.add_argument("--n_cov_states", type=int, default=1024,
                        help="val states for the whitening covariance")
    parser.add_argument("--compute_dyn_jac",
                        type=lambda s: s.lower() not in {"0", "false", "no"},
                        default=False,
                        help="Compute time-varying Jacobian metrics (jac_dyn_*). "
                             "Default: False. The fixed-Jacobian variant is "
                             "empirically the stronger MPC predictor on lander "
                             "and dynamic Jacobians ~double runtime.")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    wm_config = config.get('world_model', {})
    action_dim = wm_config.get('action_dim', 2)             # Reacher: 2 continuous
    sequence_length = wm_config.get('sequence_length', 30)
    dataset_seq_offset = wm_config.get('dataset_seq_offset', 5)
    batch_size = wm_config.get('batch_size', 64)
    beta_kl = wm_config.get('beta_kl', 0.5)

    loss_w = wm_config.get('loss_weights', {})
    loss_weights = (
        loss_w.get('reconstruction', 1.0),
        loss_w.get('reward', 1.2),
        loss_w.get('kl', 1.0),
    )

    capacity = wm_config.get('capacity', {})
    latent_dim = capacity.get('latent_dim', 16)
    hidden_dim = capacity.get('hidden_dim', 256)
    mlp_hidden_dim = capacity.get('mlp_hidden_dim', hidden_dim)
    gru_num_layers = int(capacity.get('gru_num_layers', 1))

    metrics_cfg = config.get('metrics', {})
    warmup_steps = metrics_cfg.get('warmup_steps', 5)
    planning_horizon = metrics_cfg.get('planning_horizon', 25)

    # --- Load validation data once ---
    val_dataset = SequenceDataset(
        args.val_dataset, sequence_length, action_dim,
        random_start=False, dataset_seq_offset=dataset_seq_offset)
    val_shuffle_gen = torch.Generator().manual_seed(args.seed)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                generator=val_shuffle_gen)
    obs_dim = val_dataset.obs_dim
    print("Validation dataset: {} sequences, obs_dim={}".format(len(val_dataset), obs_dim))

    # --- Curated dataloaders for Jacobian analysis (good vs bad episodes) ---
    good_idx = val_dataset.get_filtered_indices(min_return=args.good_min_return)
    bad_idx = val_dataset.get_filtered_indices(max_return=args.bad_max_return)
    good_gen = torch.Generator().manual_seed(args.seed)
    bad_gen = torch.Generator().manual_seed(args.seed)
    jac_good_dl = DataLoader(Subset(val_dataset, good_idx),
                             batch_size=batch_size, shuffle=True, generator=good_gen)
    jac_bad_dl = DataLoader(Subset(val_dataset, bad_idx),
                            batch_size=batch_size, shuffle=True, generator=bad_gen)
    print("Jacobian curated: {} good seqs (return >= {:+.1f}), "
          "{} bad seqs (return <= {:+.1f})".format(
              len(good_idx), args.good_min_return,
              len(bad_idx), args.bad_max_return))

    # --- Create model shell (weights loaded per checkpoint) ---
    world_model = WorldModel(
        obs_dim, action_dim,
        latent_dim=latent_dim, hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers, mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)

    # --- Discover checkpoints ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, args.checkpoints_dir)
    checkpoints = []
    for f in os.listdir(ckpt_dir):
        m = re.search(r"epoch_(\d+)\.pt$", f)
        if m and f.startswith("world_model_"):
            checkpoints.append((int(m.group(1)), f))
    checkpoints.sort()

    if args.epoch_min is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e >= args.epoch_min]
    if args.epoch_max is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e <= args.epoch_max]

    if args.epoch_stride > 1:
        checkpoints = checkpoints[::args.epoch_stride]

    if not checkpoints:
        raise SystemExit(
            "ERROR: no matching checkpoints found in {}; "
            "refusing to exit cleanly on an empty sweep".format(ckpt_dir))

    print("Found {} checkpoints (epochs {}-{})".format(
        len(checkpoints), checkpoints[0][0], checkpoints[-1][0]))
    print("Jacobian states: {}, stride: {}, device: {}".format(
        args.n_jac_states, args.epoch_stride, DEVICE))

    # --- Run sweep ---
    out_path = os.path.join(script_dir, args.output)
    file_mode = "a" if args.append else "w"
    with open(out_path, file_mode, encoding="utf-8") as log:
        epoch_range = ""
        if args.epoch_min is not None or args.epoch_max is not None:
            epoch_range = ", epoch range: [{}-{}]".format(
                args.epoch_min or '*', args.epoch_max or '*')
        log.write(
            "\nReacher Metrics Evaluation Sweep\n"
            "Checkpoints: {}, Seed: {}{}\n"
            "Parameters: batch_size={}, n_jac_states={}, warmup_steps={}, "
            "planning_horizon={}, epoch_stride={}, shuffle=True, "
            "compute_dyn_jac={}\n"
            "Jacobian sampling: good_min_return={:+.1f} ({} seqs), "
            "bad_max_return={:+.1f} ({} seqs)\n"
            "Device: {}\n"
            "Started: {}\n\n".format(
                len(checkpoints), args.seed, epoch_range,
                batch_size, args.n_jac_states, warmup_steps,
                planning_horizon, args.epoch_stride,
                args.compute_dyn_jac,
                args.good_min_return, len(good_idx),
                args.bad_max_return, len(bad_idx),
                DEVICE,
                time.strftime('%Y-%m-%d %H:%M:%S')))
        log.flush()

        total_t0 = time.time()
        for i, (epoch, ckpt_name) in enumerate(checkpoints):
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            # weights_only kwarg only exists in torch >= 1.13; the project's
            # current env is 1.12.1, so omit it. (train_models.py uses it but
            # this run never exercised the load branch -- training started fresh.)
            world_model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE))
            world_model.eval()

            t0 = time.time()
            metrics = {}

            whiten = None
            if args.whiten:
                rng_state = torch.get_rng_state()
                whiten = state_whiteners(world_model, val_dataset,
                                         n_states=args.n_cov_states,
                                         warmup_steps=warmup_steps,
                                         batch_size=batch_size, seed=args.seed)
                torch.set_rng_state(rng_state)

            # 1. Validation loss decomposition
            val = compute_val_metrics(world_model, val_dataloader, beta_kl,
                                      loss_weights)
            metrics.update(val)

            # 2. Multi-step rollout
            ms = compute_multistep_rollout(world_model, val_dataloader,
                                           warmup_steps=warmup_steps,
                                           max_horizon=planning_horizon)
            metrics.update(ms)

            # 3. Jacobian-based metrics (fixed linearization) -- curated good
            jac = compute_jacobian_metrics(world_model, jac_good_dl,
                                           n_states=args.n_jac_states,
                                           horizon=planning_horizon,
                                           warmup_steps=warmup_steps,
                                           whiten=whiten)
            metrics.update(jac)

            # 3b. Time-varying Jacobian metrics (optional)
            if args.compute_dyn_jac:
                jac_dyn = compute_jacobian_metrics_tv(
                    world_model, jac_good_dl,
                    n_states=args.n_jac_states,
                    horizon=planning_horizon,
                    warmup_steps=warmup_steps)
                metrics.update(jac_dyn)

            # 3c. Jacobian ROF on bad episodes
            if len(bad_idx) > 0:
                jac_bad = compute_jacobian_metrics(
                    world_model, jac_bad_dl,
                    n_states=args.n_jac_states,
                    horizon=planning_horizon,
                    warmup_steps=warmup_steps,
                    whiten=whiten)
                metrics['jac_rof_bad'] = jac_bad.get('jac_rof', float('nan'))
                for name in (whiten or {}):
                    metrics[f'jac_rof_{name}_bad'] = jac_bad.get(f'jac_rof_{name}',
                                                                 float('nan'))
                if args.compute_dyn_jac:
                    jac_dyn_bad = compute_jacobian_metrics_tv(
                        world_model, jac_bad_dl,
                        n_states=args.n_jac_states,
                        horizon=planning_horizon,
                        warmup_steps=warmup_steps)
                    metrics['jac_dyn_rof_bad'] = jac_dyn_bad.get(
                        'jac_dyn_rof', float('nan'))
                else:
                    metrics['jac_dyn_rof_bad'] = float('nan')
            else:
                metrics['jac_rof_bad'] = float('nan')
                metrics['jac_dyn_rof_bad'] = float('nan')

            # 4. Empirical C, O, L
            emp = compute_control_metrics(world_model, val_dataloader,
                                          max_states=256,
                                          seed=args.seed)
            metrics['emp_C'] = emp['ctrl_mean']
            metrics['emp_O'] = emp['obs_mean']
            # lip_max -- empirically more informative than lip_mean
            # (overfitting -> localized sensitivity blowups, captured by max).
            metrics['emp_L'] = emp['lip_max']

            elapsed = time.time() - t0
            header = "=== Epoch {} === [{}/{}] ({:.1f}s)\n  ".format(
                epoch, i + 1, len(checkpoints), elapsed)
            body = format_metrics(metrics)
            entry = header + body + "\n\n"

            print(entry, end="")
            log.write(entry)
            log.flush()

        total_elapsed = time.time() - total_t0
        footer = "Completed: {} (total: {:.1f} min)\n".format(
            time.strftime('%Y-%m-%d %H:%M:%S'), total_elapsed / 60)
        log.write(footer)
        print(footer)

    print("Results saved to {}".format(out_path))


if __name__ == "__main__":
    main()
