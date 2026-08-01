# V7 — seed 12345 continuous (May 2026, old GPU)

World-model checkpoints from Nikolai's archived V7 run
(`LunarLander_RSSM/archive/V7_good_but_worseThanV4_05212026`).

| Field | Value |
|---|---|
| Train seed | 12345 |
| Protocol | Single continuous 500-epoch run (no resume) |
| Data | Human `lunarlander_{train,val}_dataset.npz` |
| Config | V4 KL (`kl_divergence(post, prior)`); `latent_dim=16`, `beta_kl=0.5`, `reward=1.2` |
| Checkpoints | 100 files, epochs 5–500, stride 5 |
| MPC outcome | Severe late collapse (MA-7 final-10 ~−34) |

Companion logs:
- MPC sweep: `results/mpc_v7_12345_continuous.txt` (100 ckpts × 20 eps, eval seed 12345)
- Train log: in Nikolai's archive (`train_worldmodel_logs.txt`), not committed here

Usage:

```powershell
python eval_metrics.py --checkpoints_dir wm_checkpoints/v7_12345_continuous --seed 12345 ...
python eval_mpc.py --checkpoints_dir wm_checkpoints/v7_12345_continuous --seed 12345 ...
```
