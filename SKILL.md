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
2. Optimize resource allocation: CPU per job = total_cores / gpu_count. For a 32-core 8-GPU cluster, 2-GPU jobs use 8 cores and 32GB RAM.
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

For YOLO OBB training: 2 GPU + batch 32 is the recommended default for optimal throughput (~40% faster wall time than 1 GPU). Use 1 GPU + batch 16 for quick ablation experiments.

## Routing

| Intent | Action |
|---|---|
| Configure or diagnose access | Use `doctor`. |
| Upload code or configuration | Use `sync`. |
| Prepare environment, data, or weights | Pre-download weights locally → SCP to cluster. Do NOT put downloads in job scripts. |
| Run a training job | Render → review → `submit` → **immediately begin monitoring (see below)**. |
| Run independent GPU trials | Render a `mode: parallel` experiment with unique `trials`; one task and one GPU per trial. |
| Validate a trained model | Use an experiment `validation_command` after the best-model path is verified. |
| Monitor a running job | Use `status JOB_ID` to poll state; tail `.out`/`.err` logs via SSH. See monitoring protocol. |
| Inspect, cancel, or read logs | Use `status` or `cancel`. |
| Download artifacts | Use `fetch`. Remote deletion requires policy opt-in and exact confirmation. |
| Compare metrics | Use `analyze`. |
| Resolve failures | Follow the failure analysis workflow: collect logs → classify → fix → re-submit → monitor. |

## Post-Submit Monitoring (MANDATORY)

After EVERY `submit`, you MUST immediately begin automatic monitoring. Do NOT just print the job ID and stop — the job may queue, fail fast, or fake-complete. You are responsible for following the job to genuine completion.

### Monitoring Protocol

1. **Submit** the job and record the job ID.
2. **Poll interval**: run `status JOB_ID` **every 1 minute** during the critical window (first 5 minutes or until the first epoch/iteration completes, whichever is longer). After the first epoch completes successfully, relax to **every 5 minutes**.
3. **Watch for state transitions**:
   - `PENDING` → note the `Reason` field (Priority, Resources, Partition, AssocGrp*). Report to user if queued > 2 min.
   - `RUNNING` → immediately check `.out` and `.err` logs on the login node for the first lines.
   - `COMPLETED` → proceed to completion verification (below).
   - `FAILED` / `CANCELLED` / `TIMEOUT` → proceed to failure analysis (below).
4. **First-iteration gate (critical window)**: Once the job enters RUNNING, tail the `.out` log. You MUST confirm that the first training iteration/epoch has started and completed without error. Most configuration bugs, OOMs, and import errors surface here. Keep the 1-minute poll interval until the first epoch finishes. If the job dies before the first iteration finishes, it is a **fast-fail** — proceed immediately to failure analysis.
5. **Do NOT assume success from Slurm state alone.** A job can reach `COMPLETED` with exit code 0 while the training itself crashed silently.

### Job States and Meanings

| Slurm State | Meaning | Action |
|---|---|---|
| `PENDING` | Waiting for resources | Check Reason; report to user if prolonged |
| `RUNNING` | Executing on compute node | Tail logs; verify first iteration |
| `COMPLETED` | Process exited with code 0 | **Verify training actually finished** (see below) |
| `FAILED` | Process exited with non-zero code | Analyze logs → fix → re-submit |
| `TIMEOUT` | Exceeded `--time` limit | Increase time_limit or checkpoint/resume |
| `CANCELLED` | Manually cancelled or by admin | Confirm intent; re-submit if needed |
| `NODE_FAIL` | Compute node crashed | Re-submit; report to admin if repeated |

## Real Completion vs Fake Completion (假完成)

**Slurm `COMPLETED` does NOT mean the training succeeded.** It only means the shell process exited with code 0. This distinction is critical.

### Real Completion (真完成)

All of these must be true:
- Slurm state is `COMPLETED` with exit code 0.
- The `.out` log contains a completion marker (e.g., "N epochs completed", "Training complete", or the framework's final summary line).
- `results.csv` (or equivalent metrics file) exists in the output directory and has the expected number of rows (≈ number of epochs).
- `best.pt` (or equivalent checkpoint) exists and has a plausible file size (> 1 MB for most models).
- The `.err` log contains no Python tracebacks or CUDA errors.

### Fake Completion (假完成 — Job Reports COMPLETED but Training Failed)

Common patterns:
- **Silent Python crash**: The script hit an error after a few epochs, but `set -e` wasn't triggered because the training framework caught the exception and exited 0. The `.out` log ends abruptly without a completion marker.
- **Partial training**: `results.csv` exists but has fewer rows than the configured epochs. The training stopped early without raising an error.
- **Corrupted output**: `best.pt` is truncated (e.g., filesystem full, quota exceeded). Check file size.
- **NaN/inf metrics**: Training diverged; metrics are invalid. The script exited 0 but the model is unusable.
- **Data exhaustion**: Dataset too small for the configured epochs; training loop silently terminated.

### Completion Verification Checklist

After every `COMPLETED` job, run these checks before reporting success:

```text
1. grep for completion marker in .out log (e.g. "epochs completed", "results saved")
2. Count rows in results.csv — must equal configured epochs (or epochs - patience for early stop)
3. Check best.pt file size > 1 MB
4. grep for "Traceback\|Error\|Killed\|OOM\|CUBLAS\|cuDNN\|SIGKILL" in .err log
5. Run `analyze` on results.csv — verify metrics are plausible (mAP50 > 0, loss decreasing)
```

If ANY check fails, the job is a **fake completion**. Treat it as a failure: analyze logs, fix the root cause, re-render, and re-submit.

## Failure Analysis & Auto-Recovery

When a job fails (FAILED, TIMEOUT, or fake-complete), you MUST NOT just report the error. You MUST analyze the logs, classify the failure, and attempt to fix and re-submit.

### Step 1: Collect Logs

```powershell
# Fetch only the log files (not the entire output directory)
python scripts/slurm_train.py --cluster config/cluster.json fetch --remote-path logs/<job-name>-<jobid>.out --destination .\logs
python scripts/slurm_train.py --cluster config/cluster.json fetch --remote-path logs/<job-name>-<jobid>.err --destination .\logs
```

Or read logs directly via SSH:
```bash
ssh <host> "cat <workspace>/logs/<job-name>-<jobid>.out"
ssh <host> "cat <workspace>/logs/<job-name>-<jobid>.err"
```

### Step 2: Classify and Fix

| Log Signature | Failure Type | Auto-Fix |
|---|---|---|
| `ModuleNotFoundError:`, `ImportError:` | Missing Python package | Install on login node (conda/pip); re-submit |
| `FileNotFoundError: data.yaml`, `No such file` | Wrong dataset path | Fix `inputs.data` in experiment config; re-sync; re-submit |
| `CUDA out of memory`, `RuntimeError: CUDA error` | OOM | Halve batch size in experiment config; re-render; re-submit |
| `PytorchStreamReader failed`, `Failed to read` | Corrupt weight file | Re-download weights with `md5sum` verification; re-submit |
| `Name or service not known`, `Connection refused` | Network call on compute node | Move download/API call to login node; re-sync; re-submit |
| `Killed`, `SIGKILL`, `oom-killer` | System OOM (RAM, not GPU) | Reduce `--mem` request or reduce dataloader workers; re-render; re-submit |
| Traceback in first 30 seconds | Syntax/config error | Fix the code; run locally first to validate; re-sync; re-submit |
| `CUBLAS_STATUS_*`, `cuDNN error` | CUDA/cuDNN version mismatch | Check CUDA version on compute node; adjust conda env; re-submit |
| No error but training stopped at epoch N | Fake completion (see above) | Read .out log around the stop point; fix root cause; re-submit |
| `DUE TO TIME LIMIT` | Job timeout | Increase `time_limit` or reduce epochs; re-render; re-submit |

### Step 3: Re-submit Protocol

1. Fix the root cause (edit experiment config, code, or cluster config).
2. If code changed: `sync` to upload changes to the login node.
3. If config changed: `render` a new script.
4. `submit` the new script.
5. **Begin monitoring again from step 1 of the monitoring protocol.**

### Max Retry Policy

- **Syntax/config errors**: Fix and retry immediately (up to 3 attempts).
- **OOM errors**: Reduce batch size by 50% each retry (up to 3 reductions). If still OOM at batch=1, report to user.
- **Data/weight file errors**: Fix paths or re-download (up to 3 attempts). If still failing, report to user.
- **Timeout**: Increase time by 50% each retry (up to 2 increases). If still timing out, suggest checkpoint/resume strategy.
- **Node failures**: Re-submit unchanged (up to 3 attempts across different nodes).
- After **3 consecutive failures of the same type**, STOP and report to the user with a summary of what was tried.

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
| Job COMPLETED but training truncated | Fake completion (framework swallowed error) | Verify `results.csv` row count, check `.out` for tracebacks, follow completion verification checklist |
| results.csv has fewer rows than epochs | Training stopped early without error | Check `.err` for hidden tracebacks, check disk quota, verify data integrity |
| `CUBLAS_STATUS_*` | CUDA/cuDNN version mismatch | Check `nvcc --version` vs installed cudatoolkit; align env |

## Job Template

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:<N>
#SBATCH --gres-flags=enforce-binding
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
    device='0,1', workers=<CPU>, project='<PROJECT>', name='train',
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
- All GPU jobs MUST use `#SBATCH --gres-flags=enforce-binding` to guarantee exclusive GPU access and prevent other tasks from sharing the allocated GPUs.

See `README.md` for the complete setup and workflow reference.
