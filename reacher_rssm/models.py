# ===========================================================================
# Reacher RSSM - Dreamer-style world model for Reacher-v5 (continuous actions)
#
# Differences from the lander models.py:
#   - WorldModel decoder: single 10-D obs head (no physics/contact split,
#     no done head). Reacher-v5 has no discrete contact flags and no
#     early termination.
#   - RSSM and reward head are unchanged in structure; the action_dim arg
#     is now 2 (continuous torques) instead of 4 (one-hot discrete).
#
# Actor / Critic are NOT defined here yet. They will be added when the
# actor-critic phase is ported (Gaussian, tanh-squashed policy; see
# DESIGN.md section 7).
# ===========================================================================

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Recurrent State Space Model (RSSM).
#
# Identical in structure to the lander RSSM. The only thing that changes
# between environments is the action_dim used to size the GRU input.
# Kept byte-for-byte aligned with the lander to keep the metrics pipeline
# (eval_metrics.py / analyze_metrics.py) directly applicable.
# ---------------------------------------------------------------------------
class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=16,
        hidden_dim=256,
        gru_num_layers=1,
        mlp_hidden_dim=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        if self.gru_num_layers < 1:
            raise ValueError(f"gru_num_layers must be >= 1, got {self.gru_num_layers}")

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
        )

        if self.gru_num_layers == 1:
            self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        else:
            self.gru_layers = nn.ModuleList(
                [nn.GRUCell(latent_dim + action_dim, hidden_dim)]
                + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(self.gru_num_layers - 1)]
            )

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * latent_dim),
        )

        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * latent_dim),
        )

    def init_hidden(self, batch_size, device):
        if self.gru_num_layers == 1:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return torch.zeros(batch_size, self.gru_num_layers, self.hidden_dim, device=device)

    def top_hidden(self, h):
        return h[:, -1, :] if h.dim() == 3 else h

    def update_hidden(self, h, z_prev, action_prev):
        gru_in = torch.cat([z_prev, action_prev], dim=-1)

        if self.gru_num_layers == 1:
            if h.dim() == 3:
                h = h[:, 0, :]
            return self.gru(gru_in, h)

        if h.dim() == 2:
            h_prev = torch.zeros(
                h.size(0), self.gru_num_layers, self.hidden_dim, device=h.device, dtype=h.dtype
            )
            h_prev[:, 0, :] = h
        else:
            h_prev = h

        states = []
        layer_in = gru_in
        for layer_idx, cell in enumerate(self.gru_layers):
            h_i = cell(layer_in, h_prev[:, layer_idx, :])
            states.append(h_i)
            layer_in = h_i
        return torch.stack(states, dim=1)

    def prior(self, h):
        out = self.prior_net(self.top_hidden(h))
        mean, logstd = out.chunk(2, dim=-1)
        logstd = logstd.clamp(-5, 2)
        return mean, logstd

    def posterior(self, h, obs):
        obs_enc = self.obs_encoder(obs)
        post_in = torch.cat([self.top_hidden(h), obs_enc], dim=-1)
        out = self.post_net(post_in)
        mean, logstd = out.chunk(2, dim=-1)
        logstd = logstd.clamp(-5, 2)
        return mean, logstd

    def sample_latent(self, mean, logstd):
        std = torch.exp(logstd)
        return mean + std * torch.randn_like(std)

    def step(self, h, z_prev, action_prev, obs):
        h_t = self.update_hidden(h, z_prev, action_prev)
        mean_prior, logstd_prior = self.prior(h_t)
        mean_post, logstd_post = self.posterior(h_t, obs)
        z_t = self.sample_latent(mean_post, logstd_post)
        return h_t, z_t, mean_post, logstd_post, mean_prior, logstd_prior


# ---------------------------------------------------------------------------
# World model.
#
# Decoder topology vs. the lander:
#   lander:   physics_head (6) + contact_head (2 BCE) + done_head (1 BCE)
#   reacher:  angle-decoded obs_head -- see below.
#
# Observation layout (Reacher-v5; see ik_controller.py and DESIGN.md):
#   obs[0] = cos(theta_0)         obs[5] = target_y
#   obs[1] = cos(theta_1)         obs[6] = theta_dot_0
#   obs[2] = sin(theta_0)         obs[7] = theta_dot_1
#   obs[3] = sin(theta_1)         obs[8] = ftip_x - target_x
#   obs[4] = target_x             obs[9] = ftip_y - target_y
#
# The first four dims are non-independent: a valid observation MUST satisfy
# cos(theta)^2 + sin(theta)^2 = 1, and each value must lie in [-1, 1].
# A naive Linear obs_head violates both constraints and the diagnostic showed
# catastrophic prior-rollout drift in these dims.
#
# Fix: decode the TWO underlying angles directly. The head outputs 8 raw
# scalars [theta_hat_0, theta_hat_1, target_x, target_y, theta_dot_0,
# theta_dot_1, dx, dy]; we then construct the 10-D observation by computing
# (cos, sin) of the predicted angles. This makes the unit-circle constraint
# exact by construction, regardless of how far theta_hat may wander.
#
# Reward head is unchanged.
# ---------------------------------------------------------------------------
# Reacher-v5 obs structure: 2 hidden angle scalars + 6 "other" scalars =
# 8 internal head outputs that produce a 10-D observation.
NUM_ANGLES = 2
NUM_OTHER  = 6
HEAD_OUT_DIM = NUM_ANGLES + NUM_OTHER


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=16,
        hidden_dim=256,
        gru_num_layers=1,
        mlp_hidden_dim=None,
    ):
        super().__init__()
        if obs_dim != NUM_ANGLES * 2 + NUM_OTHER:
            raise ValueError(
                f"WorldModel currently assumes Reacher-v5 obs layout "
                f"({NUM_ANGLES * 2 + NUM_OTHER}-D); got obs_dim={obs_dim}."
            )
        self.obs_dim = obs_dim
        self.mlp_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(hidden_dim)

        self.rssm = RSSM(
            obs_dim,
            action_dim,
            latent_dim,
            hidden_dim,
            gru_num_layers=gru_num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )

        dec_in_dim = latent_dim + hidden_dim
        self.decoder_backbone = nn.Sequential(
            nn.Linear(dec_in_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
        )
        self.obs_head = nn.Linear(self.mlp_hidden_dim, HEAD_OUT_DIM)

        self.reward_head = nn.Sequential(
            nn.Linear(dec_in_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def decode_obs(self, h, z):
        """Return (B, 10) observation with cos/sin invariants enforced.

        The head outputs 8 raw scalars: 2 hidden joint angles and 6 'other'
        scalars (target_xy, joint velocities, vector-to-target). The 10-D
        Reacher-v5 obs is reconstructed by analytically applying cos/sin
        to the angles. Since cos/sin are 2pi-periodic and bounded in
        [-1, 1] by construction, the predicted angle scalars are free to
        wander (no wrapping needed) and the unit-circle constraint is
        satisfied exactly.
        """
        x = torch.cat([self.rssm.top_hidden(h), z], dim=-1)
        feat = self.decoder_backbone(x)
        raw = self.obs_head(feat)                        # (B, 8)
        theta = raw[..., :NUM_ANGLES]                    # (B, 2)
        other = raw[..., NUM_ANGLES:]                    # (B, 6)
        return torch.cat([torch.cos(theta), torch.sin(theta), other], dim=-1)

    def reconstruct_obs(self, h, z):
        return self.decode_obs(h, z)

    def predict_reward(self, h, z):
        return self.reward_head(torch.cat([self.rssm.top_hidden(h), z], dim=-1)).squeeze(-1)
