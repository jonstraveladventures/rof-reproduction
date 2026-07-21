# ===========================================================================
# Reacher RSSM - World-model training (continuous actions, Reacher-v5).
#
# Differences from the lander train_models.py:
#   - Continuous 2-D float actions: no one-hot encoding, feed actions
#     directly into the RSSM GRU.
#   - 10-D obs decoded with the angle-decoded obs_head defined in models.py
#     (predict 2 hidden joint angles, expand to cos/sin analytically).
#   - No done head and no done loss (Reacher episodes are fixed-length).
#   - Reconstruction loss is PER-DIM MEAN MSE (was per-dim sum), so reward
#     is no longer ~10x under-weighted relative to obs.
#   - KL is balanced (DreamerV2/V3 style):
#       L_kl = alpha * KL(sg(post)||prior) + (1-alpha)* KL(post||sg(prior))
#     with alpha=kl_alpha (default 0.8). Replaces the symmetric KL the
#     lander script used.
#   - Validation produces both 1-step posterior metrics and an open-loop
#     rollout MAE (warmup + prior-only rollout for `planning_horizon`
#     steps); the rollout metric is the actual signal we care about for
#     CEM-MPC quality.
#   - PyTorch 2.11 API: torch.load(..., weights_only=True).
#
# This file ONLY implements --phase world_model. The actor-critic phase
# (continuous Gaussian policy) is ported in a separate step; see DESIGN.md
# section 7.
# ===========================================================================

import argparse
import datetime
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from models import WorldModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Utilities (same shape as lander script for consistency).
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloader_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_message(message, log_path=None):
    print(message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def get_latest_checkpoint(model_name, directory=None):
    if directory is None:
        directory = CHECKPOINT_DIR
    pattern = os.path.join(directory, f"{model_name}_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def save_checkpoint_with_timestamp(model, model_name, epoch, directory=None, log_path=None):
    if directory is None:
        directory = CHECKPOINT_DIR
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{model_name}_{timestamp}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), filename)
    log_message(f"Saved checkpoint: {filename}", log_path)
    return filename


# ---------------------------------------------------------------------------
# SequenceDataset for sequence-based training.
#
# Schema for Reacher (see DESIGN.md section 4):
#   obs           float32 (N, 10)
#   actions       float32 (N, 2)        -- continuous, in [-1, 1]
#   rewards       float32 (N,)
#   next_obs      float32 (N, 10)
#   dones         int64   (N,)          -- 0 except 1 at step 99 of each ep
#   ep_index      int64   (N,)
#   step_index    int64   (N,)
#   episode_seed  int64   (N,)
#   episode_class int64   (N,)          -- 0/1/2; ignored here
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Samples sequences from real episodes for world-model training."""

    def __init__(self, path, sequence_length, action_dim, random_start=True,
                 dataset_seq_offset=20):
        data = np.load(path)
        self.obs        = data["obs"].astype(np.float32)
        self.actions    = data["actions"].astype(np.float32)        # continuous (N, 2)
        self.rewards    = data["rewards"].astype(np.float32)
        self.next_obs   = data["next_obs"].astype(np.float32)
        self.dones      = data["dones"].astype(np.int64)
        self.ep_index   = data["ep_index"].astype(np.int64)
        self.step_index = data["step_index"].astype(np.int64)
        self.sequence_length = int(sequence_length)
        self.random_start = bool(random_start)
        self.dataset_seq_offset = max(1, int(dataset_seq_offset))
        self.obs_dim = int(self.obs.shape[1])

        # action_dim is now the *continuous* action dimension.  Sanity check it.
        self.action_dim = int(action_dim)
        if self.actions.ndim != 2 or self.actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Dataset actions shape {self.actions.shape} does not match "
                f"action_dim={self.action_dim} (expected (N, {self.action_dim}))."
            )

        # Group rows by episode, sort by step_index, generate start positions.
        self.episode_ids = np.unique(self.ep_index)
        self.episode_indices = []
        self.episode_returns = []
        self.start_positions = []
        for ep_id in self.episode_ids:
            data_idxs = np.where(self.ep_index == ep_id)[0]
            if data_idxs.size == 0:
                continue
            order = np.argsort(self.step_index[data_idxs], kind="stable")
            data_idxs = data_idxs[order]
            ep_pos = len(self.episode_indices)
            self.episode_indices.append(data_idxs)
            self.episode_returns.append(float(self.rewards[data_idxs].sum()))
            for pos in range(0, len(data_idxs), self.dataset_seq_offset):
                self.start_positions.append((ep_pos, pos))
        self.episode_returns = np.array(self.episode_returns)
        self.seq_ep_return = np.array([
            self.episode_returns[ep_pos]
            for ep_pos, _ in self.start_positions
        ])

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        if self.random_start:
            ep_pos, start = self.start_positions[np.random.randint(len(self.start_positions))]
        else:
            ep_pos, start = self.start_positions[idx]
        data_idxs = self.episode_indices[ep_pos]

        obs_seq      = np.zeros((self.sequence_length, self.obs_dim),    dtype=np.float32)
        actions_seq  = np.zeros((self.sequence_length, self.action_dim), dtype=np.float32)
        rewards_seq  = np.zeros((self.sequence_length,),                  dtype=np.float32)
        next_obs_seq = np.zeros((self.sequence_length, self.obs_dim),    dtype=np.float32)
        dones_seq    = np.zeros((self.sequence_length,),                  dtype=np.int64)
        mask         = np.zeros((self.sequence_length,),                  dtype=np.float32)

        max_len = len(data_idxs) - start
        for t in range(self.sequence_length):
            if t >= max_len:
                break
            index_t = data_idxs[start + t]
            obs_seq[t]      = self.obs[index_t]
            actions_seq[t]  = self.actions[index_t]
            rewards_seq[t]  = self.rewards[index_t]
            next_obs_seq[t] = self.next_obs[index_t]
            dones_seq[t]    = self.dones[index_t]
            mask[t]         = 1.0
            if self.dones[index_t] == 1:
                break

        return obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask

    def get_filtered_indices(self, min_return=None, max_return=None):
        """Return sequence indices whose source episode return is in [min, max]."""
        mask = np.ones(len(self.start_positions), dtype=bool)
        if min_return is not None:
            mask &= self.seq_ep_return >= min_return
        if max_return is not None:
            mask &= self.seq_ep_return <= max_return
        return np.where(mask)[0]


# ---------------------------------------------------------------------------
# KL divergence between two diagonal Gaussians (same as lander).
# ---------------------------------------------------------------------------
def kl_divergence(mean_q, logstd_q, mean_p, logstd_p):
    var_q = torch.exp(2 * logstd_q)
    var_p = torch.exp(2 * logstd_p)
    return 0.5 * (
        (var_q + (mean_q - mean_p) ** 2) / var_p
        - 1.0
        + 2 * (logstd_p - logstd_q)
    ).sum(-1)


# ---------------------------------------------------------------------------
# DreamerV2 / V3 KL balancing.
#
#   L_kl = alpha   * KL(sg(post)  || prior)        # "dynamics loss": prior follows post
#        + (1-alpha)* KL( post     || sg(prior))   # "representation loss": post follows prior
#
# alpha = 0.8 is the standard Dreamer setting: prior chases the posterior more
# strongly than the posterior chases the prior. This matters specifically for
# open-loop CEM rollouts, which run the prior alone -- a free-running posterior
# (the symmetric KL setting) lets the posterior drift away from a useful prior.
# ---------------------------------------------------------------------------
def kl_balanced_divergence(mean_post, logstd_post,
                           mean_prior, logstd_prior, alpha=0.8):
    kl_dyn = kl_divergence(
        mean_post.detach(), logstd_post.detach(),
        mean_prior, logstd_prior,
    )                                # gradient flows to prior only
    kl_rep = kl_divergence(
        mean_post, logstd_post,
        mean_prior.detach(), logstd_prior.detach(),
    )                                # gradient flows to posterior only
    return alpha * kl_dyn + (1.0 - alpha) * kl_rep


# ---------------------------------------------------------------------------
# World-model training loop.
# ---------------------------------------------------------------------------
def train_world_model(world_model, train_dataloader, val_dataloader,
                      epochs=10, start_epoch=0, checkpoint_freq=100,
                      val_freq=10, lr=3e-4, beta_kl=1.0, kl_alpha=0.8,
                      loss_weights=(1.0, 1.0, 1.0),     # (recon, reward, kl)
                      rollout_warmup=5, rollout_horizon=25,
                      log_path=None):

    world_model.train()
    opt = optim.AdamW(world_model.parameters(), lr=lr)

    if start_epoch >= epochs:
        log_message(f"Start epoch {start_epoch} is >= total epochs {epochs}; nothing to train.", log_path)
        return

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = 0.0
        for batch in train_dataloader:
            obs_seq, actions_seq, rewards_seq, next_obs_seq, _dones_seq, mask = batch
            obs_seq      = obs_seq.to(DEVICE)
            actions_seq  = actions_seq.to(DEVICE)       # (B, T, action_dim) float
            rewards_seq  = rewards_seq.to(DEVICE)
            next_obs_seq = next_obs_seq.to(DEVICE)
            mask         = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_init = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew   = 0.0
            sum_kl    = 0.0
            total_mask = mask.sum().clamp_min(1.0)

            # Bootstrap: encode obs_seq[:, 0] to get initial RSSM state (h_0, z_0).
            h = world_model.rssm.update_hidden(h, z, a_init)
            mean_post_0, logstd_post_0 = world_model.rssm.posterior(h, obs_seq[:, 0])
            z = world_model.rssm.sample_latent(mean_post_0, logstd_post_0)

            # Transition loop: (s_t, a_t) -> s_{t+1}, supervised with next_obs[t].
            for t in range(seq_len):
                a_t = actions_seq[:, t]                # continuous, no one-hot

                h = world_model.rssm.update_hidden(h, z, a_t)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post,  logstd_post  = world_model.rssm.posterior(h, next_obs_seq[:, t])
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred    = world_model.decode_obs(h, z)            # (B, 10)
                reward_pred = world_model.predict_reward(h, z)        # (B,)

                # Per-dim mean MSE (was sum). Item 3 of the WM-fix plan: this
                # rebalances the loss so reward (a single scalar per step) is
                # not ~10x under-weighted relative to a 10-D-summed obs recon.
                recon_loss  = (obs_pred - next_obs_seq[:, t]).pow(2).mean(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                kl          = kl_balanced_divergence(
                    mean_post, logstd_post, mean_prior, logstd_prior,
                    alpha=kl_alpha,
                )

                mask_t = mask[:, t]
                sum_recon += (recon_loss  * mask_t).sum()
                sum_rew   += (reward_loss * mask_t).sum()
                sum_kl    += (kl          * mask_t).sum()

            loss = (
                loss_weights[0] * sum_recon
                + loss_weights[1] * sum_rew
                + loss_weights[2] * beta_kl * sum_kl
            ) / total_mask

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        if epoch % val_freq == 0:
            val_metrics = validate_world_model(
                world_model, val_dataloader,
                beta_kl=beta_kl, kl_alpha=kl_alpha, loss_weights=loss_weights,
                rollout_warmup=rollout_warmup, rollout_horizon=rollout_horizon,
            )
            log_message(
                f"[WorldModel] Epoch {epoch}, train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}",
                log_path,
            )
            log_message(
                f"  1-step    - obs_mae={val_metrics['obs_mae']:.4f} "
                f"obs_rmse={val_metrics['obs_rmse']:.4f} "
                f"rew_mae={val_metrics['reward_mae']:.4f} "
                f"rew_rmse={val_metrics['reward_rmse']:.4f}",
                log_path,
            )
            log_message(
                f"  rollout(warmup={rollout_warmup}, h={rollout_horizon}) - "
                f"obs_mae={val_metrics['obs_mae_rollout']:.4f} "
                f"rew_mae={val_metrics['reward_mae_rollout']:.4f} "
                f"obs_mae@H={val_metrics['obs_mae_rollout_final']:.4f} "
                f"rew_mae@H={val_metrics['reward_mae_rollout_final']:.4f}",
                log_path,
            )
        else:
            log_message(f"[WorldModel] Epoch {epoch}, train_loss={train_loss:.4f}", log_path)

        if epoch % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(world_model, "world_model", epoch, log_path=log_path)


def validate_world_model(world_model, val_dataloader, beta_kl=1.0, kl_alpha=0.8,
                         loss_weights=(1.0, 1.0, 1.0),
                         rollout_warmup=5, rollout_horizon=25):
    """Validation pass producing both 1-step posterior metrics and an open-loop
    rollout-MAE measurement.

    The rollout metric (Item 4 of the WM-fix plan) is what CEM-MPC actually
    needs: warm the posterior for a few steps, then run prior-only for the
    planning horizon. We compare every rollout step against ground truth and
    report both the average MAE over the rollout and the final-step MAE
    (drift at the planning horizon).
    """
    was_training = world_model.training
    world_model.eval()
    obs_dim = world_model.rssm.obs_dim

    # 1-step posterior metrics (preserved for backwards comparison).
    total_loss = 0.0
    total_obs_abs = 0.0
    total_obs_sq = 0.0
    total_reward_abs = 0.0
    total_reward_sq = 0.0
    total_mask = 0.0

    # Rollout (prior-only) metrics. We only score steps that are inside both
    # the rollout window and the data mask.
    rollout_obs_abs_per_step = np.zeros(rollout_horizon, dtype=np.float64)
    rollout_rew_abs_per_step = np.zeros(rollout_horizon, dtype=np.float64)
    rollout_count_per_step   = np.zeros(rollout_horizon, dtype=np.float64)

    with torch.no_grad():
        for batch in val_dataloader:
            obs_seq, actions_seq, rewards_seq, next_obs_seq, _dones_seq, mask = batch
            obs_seq      = obs_seq.to(DEVICE)
            actions_seq  = actions_seq.to(DEVICE)
            rewards_seq  = rewards_seq.to(DEVICE)
            next_obs_seq = next_obs_seq.to(DEVICE)
            mask         = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            # ---- Pass 1: 1-step posterior validation (matches training loop) ----
            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_init = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew   = 0.0
            sum_kl    = 0.0

            h = world_model.rssm.update_hidden(h, z, a_init)
            mean_post_0, logstd_post_0 = world_model.rssm.posterior(h, obs_seq[:, 0])
            z = world_model.rssm.sample_latent(mean_post_0, logstd_post_0)

            for t in range(seq_len):
                a_t = actions_seq[:, t]
                h = world_model.rssm.update_hidden(h, z, a_t)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post,  logstd_post  = world_model.rssm.posterior(h, next_obs_seq[:, t])
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred    = world_model.decode_obs(h, z)
                reward_pred = world_model.predict_reward(h, z)

                recon_loss  = (obs_pred - next_obs_seq[:, t]).pow(2).mean(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                kl          = kl_balanced_divergence(
                    mean_post, logstd_post, mean_prior, logstd_prior,
                    alpha=kl_alpha,
                )

                mask_t = mask[:, t]
                sum_recon += (recon_loss  * mask_t).sum()
                sum_rew   += (reward_loss * mask_t).sum()
                sum_kl    += (kl          * mask_t).sum()

                obs_diff_abs = (obs_pred - next_obs_seq[:, t]).abs().mean(-1)
                obs_diff_sq  = (obs_pred - next_obs_seq[:, t]).pow(2).mean(-1)
                total_obs_abs    += (obs_diff_abs * mask_t).sum().item()
                total_obs_sq     += (obs_diff_sq  * mask_t).sum().item()
                total_reward_abs += ((reward_pred - rewards_seq[:, t]).abs() * mask_t).sum().item()
                total_reward_sq  += ((reward_pred - rewards_seq[:, t]).pow(2) * mask_t).sum().item()

                total_mask += mask_t.sum().item()

            batch_mask = mask.sum().item()
            if batch_mask > 0:
                total_loss += (
                    loss_weights[0] * sum_recon
                    + loss_weights[1] * sum_rew
                    + loss_weights[2] * beta_kl * sum_kl
                ).item()

            # ---- Pass 2: open-loop rollout MAE (prior-only, deterministic) ----
            # Skip if the sequence is shorter than warmup + at least 1 rollout step.
            if seq_len < rollout_warmup + 1:
                continue

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z = torch.zeros(batch_size, latent_dim, device=DEVICE)

            h = world_model.rssm.update_hidden(h, z, a_init)
            mean_post_0, logstd_post_0 = world_model.rssm.posterior(h, obs_seq[:, 0])
            z = world_model.rssm.sample_latent(mean_post_0, logstd_post_0)

            # Warm posterior for `rollout_warmup` steps.
            for t in range(rollout_warmup):
                a_t = actions_seq[:, t]
                h = world_model.rssm.update_hidden(h, z, a_t)
                _mean_prior, _logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, next_obs_seq[:, t])
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

            # Open-loop rollout: prior MEAN (deterministic) for up to
            # rollout_horizon steps, comparing to ground truth.
            max_rollout = min(rollout_horizon, seq_len - rollout_warmup)
            for k in range(max_rollout):
                t = rollout_warmup + k
                a_t = actions_seq[:, t]
                h = world_model.rssm.update_hidden(h, z, a_t)
                mean_prior, _logstd_prior = world_model.rssm.prior(h)
                z = mean_prior              # deterministic open-loop

                obs_pred    = world_model.decode_obs(h, z)
                reward_pred = world_model.predict_reward(h, z)

                mask_t = mask[:, t]
                obs_diff_abs = (obs_pred - next_obs_seq[:, t]).abs().mean(-1)
                rew_diff_abs = (reward_pred - rewards_seq[:, t]).abs()

                rollout_obs_abs_per_step[k] += (obs_diff_abs * mask_t).sum().item()
                rollout_rew_abs_per_step[k] += (rew_diff_abs * mask_t).sum().item()
                rollout_count_per_step[k]   += mask_t.sum().item()

    world_model.train(was_training)

    if total_mask == 0:
        return {
            "loss": 0.0,
            "obs_mae": 0.0, "obs_rmse": 0.0,
            "reward_mae": 0.0, "reward_rmse": 0.0,
            "obs_mae_rollout": 0.0, "reward_mae_rollout": 0.0,
            "obs_mae_rollout_final": 0.0, "reward_mae_rollout_final": 0.0,
        }

    valid_steps = rollout_count_per_step > 0
    if valid_steps.any():
        obs_mae_per_step = np.where(valid_steps,
                                    rollout_obs_abs_per_step / np.maximum(rollout_count_per_step, 1.0),
                                    0.0)
        rew_mae_per_step = np.where(valid_steps,
                                    rollout_rew_abs_per_step / np.maximum(rollout_count_per_step, 1.0),
                                    0.0)
        obs_mae_rollout = float(obs_mae_per_step[valid_steps].mean())
        rew_mae_rollout = float(rew_mae_per_step[valid_steps].mean())
        last_valid = int(np.where(valid_steps)[0].max())
        obs_mae_final = float(obs_mae_per_step[last_valid])
        rew_mae_final = float(rew_mae_per_step[last_valid])
    else:
        obs_mae_rollout = rew_mae_rollout = 0.0
        obs_mae_final   = rew_mae_final   = 0.0

    return {
        "loss":                     total_loss / total_mask,
        "obs_mae":                  total_obs_abs    / total_mask,
        "obs_rmse":                 float(np.sqrt(total_obs_sq    / total_mask)),
        "reward_mae":               total_reward_abs / total_mask,
        "reward_rmse":              float(np.sqrt(total_reward_sq / total_mask)),
        "obs_mae_rollout":          obs_mae_rollout,
        "reward_mae_rollout":       rew_mae_rollout,
        "obs_mae_rollout_final":    obs_mae_final,
        "reward_mae_rollout_final": rew_mae_final,
    }


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Reacher-v5 world model from collected datasets."
    )
    parser.add_argument("--phase", choices=["world_model", "actor_critic"], required=True,
                        help="Which phase to train. Only world_model is implemented "
                             "in this file; actor_critic is added in a separate port.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--train_dataset", type=str, default="reacher_train_dataset.npz",
                        help="Path to training dataset")
    parser.add_argument("--val_dataset", type=str, default="reacher_val_dataset.npz",
                        help="Path to validation dataset (world model only)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for reproducibility")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoints and train from scratch. "
                             "Use this whenever the model architecture changes (e.g. "
                             "after the angle-decoded obs_head, old checkpoints have "
                             "incompatible obs_head shapes and will fail to load).")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: checkpoints)")
    args = parser.parse_args()

    global CHECKPOINT_DIR
    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.phase == "actor_critic":
        raise NotImplementedError(
            "actor_critic phase is not ported yet for Reacher. The continuous-action "
            "Gaussian policy is planned in a follow-up step (see DESIGN.md section 7)."
        )

    phase_config = config.get("world_model", {})
    log_path = os.path.join(".", "train_worldmodel_logs.txt")

    sequence_length     = phase_config.get("sequence_length", 30)
    dataset_seq_offset  = phase_config.get("dataset_seq_offset", 5)
    action_dim          = phase_config.get("action_dim", 2)

    train_dataset = SequenceDataset(
        args.train_dataset,
        sequence_length,
        action_dim,
        random_start=True,
        dataset_seq_offset=dataset_seq_offset,
    )
    val_dataset = SequenceDataset(
        args.val_dataset,
        sequence_length,
        action_dim,
        random_start=False,
        dataset_seq_offset=dataset_seq_offset,
    )

    batch_size = phase_config.get("batch_size", 64)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=get_dataloader_generator(args.seed),
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    obs_dim          = train_dataset.obs_dim
    capacity         = phase_config.get("capacity", {})
    latent_dim       = capacity.get("latent_dim", 16)
    hidden_dim       = capacity.get("hidden_dim", 256)
    mlp_hidden_dim   = capacity.get("mlp_hidden_dim", hidden_dim)
    gru_num_layers   = int(capacity.get("gru_num_layers", 1))

    world_model = WorldModel(
        obs_dim,
        action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)

    latest_checkpoint = None if args.fresh else get_latest_checkpoint("world_model")
    start_epoch = 0
    if latest_checkpoint:
        state = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=True)
        try:
            world_model.load_state_dict(state)
            log_message(f"Loaded world model checkpoint: {latest_checkpoint}", log_path)
            try:
                epoch_str = latest_checkpoint.split("_epoch_")[-1].split(".")[0]
                start_epoch = int(epoch_str)
            except ValueError:
                pass
        except RuntimeError as e:
            log_message(
                f"ERROR: failed to load checkpoint {latest_checkpoint}: {e}", log_path,
            )
            log_message(
                "Architecture has changed since this checkpoint was saved (e.g. after "
                "the angle-decoded obs_head). Re-run with --fresh to ignore old "
                "checkpoints and train from scratch, or move them out of "
                f"./{CHECKPOINT_DIR}/.",
                log_path,
            )
            raise
    elif (not args.fresh) and os.path.exists("world_model.pt"):
        state = torch.load("world_model.pt", map_location=DEVICE, weights_only=True)
        try:
            world_model.load_state_dict(state)
            log_message("Loaded world model from world_model.pt", log_path)
        except RuntimeError as e:
            log_message(
                f"ERROR: failed to load world_model.pt: {e}\n"
                "Re-run with --fresh to ignore it.",
                log_path,
            )
            raise

    epochs           = phase_config.get("epochs", 500)
    lr               = phase_config.get("lr", 1e-4)
    beta_kl          = phase_config.get("beta_kl", 0.5)
    kl_alpha         = phase_config.get("kl_alpha", 0.8)
    val_freq         = phase_config.get("val_freq", 5)
    checkpoint_freq  = phase_config.get("checkpoint_freq", 5)

    loss_weights     = phase_config.get("loss_weights", {})
    recon_weight     = loss_weights.get("reconstruction", 1.0)
    reward_weight    = loss_weights.get("reward", 1.2)
    kl_weight        = loss_weights.get("kl", 1.0)

    metrics_config   = config.get("metrics", {})
    rollout_warmup   = int(metrics_config.get("warmup_steps", 5))
    rollout_horizon  = int(metrics_config.get("planning_horizon", 25))

    log_message(f"Train dataset file: {args.train_dataset}", log_path)
    log_message(f"Validation dataset file: {args.val_dataset}", log_path)
    log_message(f"Train dataset episodes: {len(train_dataset.episode_ids)}", log_path)
    log_message(f"Train dataset samples/sequences: {len(train_dataset)}", log_path)
    log_message(f"Validation dataset episodes: {len(val_dataset.episode_ids)}", log_path)
    log_message(f"Validation dataset samples/sequences: {len(val_dataset)}", log_path)

    log_message(
        f"Training world model for {epochs} epochs with lr={lr}, beta_kl={beta_kl}, "
        f"kl_alpha={kl_alpha}, sequence_length={sequence_length}, "
        f"dataset_seq_offset={dataset_seq_offset}, seed={args.seed}, device={DEVICE}",
        log_path,
    )
    log_message(
        f"Model capacity: obs_dim={obs_dim}, action_dim={action_dim}, "
        f"latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
        f"mlp_hidden_dim={mlp_hidden_dim}, gru_num_layers={gru_num_layers}",
        log_path,
    )
    log_message(
        f"Loss weights: recon={recon_weight} (per-dim mean), "
        f"reward={reward_weight}, kl={kl_weight}",
        log_path,
    )
    log_message(
        f"Rollout-MAE validation: warmup={rollout_warmup}, horizon={rollout_horizon}",
        log_path,
    )

    train_world_model(
        world_model, train_dataloader, val_dataloader,
        epochs=epochs, start_epoch=start_epoch,
        checkpoint_freq=checkpoint_freq, val_freq=val_freq,
        lr=lr, beta_kl=beta_kl, kl_alpha=kl_alpha,
        loss_weights=(recon_weight, reward_weight, kl_weight),
        rollout_warmup=rollout_warmup, rollout_horizon=rollout_horizon,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
