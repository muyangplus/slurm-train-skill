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

1. Read the user's cluster config (hardware specs, partition, paths, conda env, network status).
2. Optimize resource allocation: CPU per job = total_cores / gpu_count. For a 32-core 8-GPU cluster, 1-GPU jobs use 4 cores and 16GB RAM.
3. If compute nodes are offline, pre-download all model weights locally and SCP them to the cluster `weights/` directory before submitting any job.
4. Copy `config/cluster.example.json` to a local `config/cluster.json`; do not commit it.
5. Copy an experiment example, fill in framework-specific command arrays and paths, and keep all remote paths relative to the configured workspace.
6. Run `python scripts/slurm_train.py --cluster config/cluster.json --dry-run doctor` first. Remove `--dry-run` only after the printed SSH command is correct.
7. Render and review a script before submission.

## Resource Allocation

Scale job resources based on the cluster's total hardware:

| Job Type | GPU | CPU | RAM | Max Concurrent (8-GPU 32-core) |
|----------|-----|-----|-----|-------------------------------|
| small | 1 | 4 | 16G | 8 |
| medium | 2 | 8 | 32G | 4 |
| large | 4 | 16 | 64G | 2 |

For YOLO OBB training: 1 GPU + batch 16 is sufficient for most experiments. Use 2 GPU + batch 32 for final long-training runs.

## Routing

| Intent | Action |
|---|---|
| Configure or diagnose access | Use `doctor`. |
| Upload code or configuration | Use `sync`. |
| Prepare environment, data, or weights | Pre-download weights locally → SCP to cluster. Do NOT put downloads in job scripts. |
| Run one GPU job | Render a `mode: single` experiment, review it, then use `submit`. |
| Run independent GPU trials | Render a `mode: parallel` experiment with unique `trials`; one task and one GPU per trial. |
| Validate a trained model | Use an experiment `validation_command` after the best-model path is verified. |
| Inspect, cancel, or read logs | Use `status` or `cancel`. |
| Download artifacts | Use `fetch`. Remote deletion requires policy opt-in and exact confirmation. |
| Compare metrics | Use `analyze`. |
| Resolve failures | Check local connection → login-node setup → Slurm reason/exit code → compute-node logs. |

## Common Failure Patterns

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Name or service not known` | Compute node has no internet | Pre-download weights locally → SCP to `weights/` |
| `PytorchStreamReader failed` | Corrupted weight file | Re-download with `curl -L --retry 3`, verify `md5sum` |
| CUDA OOM | Batch too large or GPU shared | Reduce batch, check `nvidia-smi` on compute node |
| Job FAILED in <1 min | Syntax error or missing import | Run script locally first, check `.err` log |
| `FileNotFoundError: data.yaml` | Wrong dataset path | Verify path exists on cluster with `ssh ... ls` |
| SSH timeout during upload | VPN instability | Use `tar -czf - \| ssh ... tar -xzf -` with `ServerAliveInterval 60` |
| `ModuleNotFoundError: custom_module` | Code not synced | SCP all custom `.py` files before submitting |

## Job Template

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:<N>
#SBATCH --cpus-per-task=<CPU>
#SBATCH --mem=<MEM>
#SBATCH --time=<TIME>
#SBATCH --output=<LOG_DIR>/<NAME>-%j.out
#SBATCH --error=<LOG_DIR>/<NAME>-%j.err

set -e
cd <WORKSPACE>

source $(conda info --base)/etc/profile.d/conda.sh
conda activate <ENV>

python -c "
from ultralytics import YOLO
model = YOLO('<WEIGHT_PATH>')
results = model.train(
    data='<DATA_YAML>', epochs=<N>, batch=<N>, imgsz=640,
    device=0, workers=<CPU>, project='<PROJECT>', name='train',
    exist_ok=True, patience=20, save=True, save_period=10, val=True,
)
rd = results.results_dict
import json, os
metrics = {
    'mAP50': float(rd.get('metrics/mAP50(B)', 0)),
    'mAP50_95': float(rd.get('metrics/mAP50-95(B)', 0)),
}
os.makedirs('<RESULT_DIR>', exist_ok=True)
json.dump(metrics, open('<RESULT_DIR>/<NAME>.json', 'w'), indent=2)
print(json.dumps(metrics, indent=2))
"
```

## Safety Rules

- Ask for the user's SSH host/alias, username, workspace, partition, account/QoS, and approved module/Conda setup. Never invent them.
- Keep passwords, tokens, private keys, host-specific values, and local `cluster.json` out of Git, output, and generated scripts.
- Validate experiment names against `[A-Za-z0-9][A-Za-z0-9._-]*`; use a unique output directory for every trial.
- Treat `--delete-remote` as destructive. It is disabled unless the cluster policy permits it and requires `--confirm` with the exact remote path.
- Do not request multiple GPUs for a single-process workload. Parallel sweeps use independent single-GPU tasks and Slurm-controlled GPU visibility.

See `README.md` for the complete setup and workflow reference.
