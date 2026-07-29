---
name: slurm-train-skill
description: Configure, validate, synchronize, submit, monitor, retrieve, and analyze training jobs on generic SLURM GPU clusters. Use for SLURM training, sbatch scripts, GPU job queues, cluster sync, result retrieval, and training metrics.
---

# SLURM Train Skill

Use this Skill to operate a training workflow across three execution boundaries:

- **Local PC**: configuration, rendering, source synchronization, artifact retrieval, and CSV analysis.
- **Login node**: directory setup, dependency and weight preparation, `sbatch`, `squeue`, `scontrol`, `sacct`, and `scancel`.
- **Compute node**: only prevalidated training or validation commands launched by SLURM.

Never assume a compute node can access the internet. By default, compute-node network access is disabled in `cluster.json`; dependency installation and weight downloads belong on the login node. Do not assume a direct SSH route to a GPU node: use SLURM allocation and a site-approved probe when GPU inspection is needed.

## First Use

1. Copy `config/cluster.example.json` to a local `config/cluster.json`; do not commit it.
2. Copy an experiment example, fill in framework-specific command arrays and paths, and keep all remote paths relative to the configured workspace.
3. Run `python scripts/slurm_train.py --cluster config/cluster.json --dry-run doctor` first. Remove `--dry-run` only after the printed SSH command is correct.
4. Render and review a script before submission:

```powershell
python scripts/slurm_train.py --cluster config/cluster.json render --experiment config/experiment.single.example.json --output rendered/baseline.slurm
```

## Routing

| Intent | Action |
|---|---|
| Configure or diagnose access | Use `doctor`. |
| Upload code or configuration | Use `sync`. |
| Prepare environment, data, or weights | Run checks on the login node. Do not put installs/downloads in a job script. |
| Run one GPU job | Render a `mode: single` experiment, review it, then use `submit`. |
| Run independent GPU trials | Render a `mode: parallel` experiment with unique `trials`; one task and one GPU per trial. |
| Validate a trained model | Use an experiment `validation_command` after the best-model path is verified. |
| Inspect, cancel, or read logs | Use `status` or `cancel`. |
| Download artifacts | Use `fetch`. Remote deletion requires policy opt-in and exact confirmation. |
| Compare metrics | Use `analyze`. |
| Resolve failures | Check the local connection, login-node setup, Slurm reason/exit code, then compute-node logs in that order. |

## Safety Rules

- Ask for the user's SSH host/alias, username, workspace, partition, account/QoS, and approved module/Conda setup. Never invent them.
- Keep passwords, tokens, private keys, host-specific values, and local `cluster.json` out of Git, output, and generated scripts.
- Validate experiment names against `[A-Za-z0-9][A-Za-z0-9._-]*`; use a unique output directory for every trial.
- Treat `--delete-remote` as destructive. It is disabled unless the cluster policy permits it and requires `--confirm` with the exact remote path.
- Do not request multiple GPUs for a single-process workload. Parallel sweeps use independent single-GPU tasks and Slurm-controlled GPU visibility.

See `README.md` for the complete setup and workflow reference.
