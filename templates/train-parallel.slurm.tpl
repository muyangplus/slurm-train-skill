#!/usr/bin/env bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --ntasks={{TRIAL_COUNT}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --gpus-per-task={{GPUS_PER_TASK}}
#SBATCH --partition={{PARTITION}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --output={{LOG_DIR}}/%x-%j-%t.out
#SBATCH --error={{LOG_DIR}}/%x-%j-%t.err
{{OPTIONAL_DIRECTIVES}}

{{COMMON}}
case "${SLURM_PROCID:?missing Slurm task id}" in
{{TRIAL_CASES}}
  *) echo "Unknown task id: ${SLURM_PROCID}" >&2; exit 2 ;;
esac
{{INPUT_CHECKS}}
srun --exclusive --ntasks=1 --gpus-per-task={{GPUS_PER_TASK}} bash -lc "$TRAIN_COMMAND"
