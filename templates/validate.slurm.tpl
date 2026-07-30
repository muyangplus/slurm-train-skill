#!/usr/bin/env bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --gpus-per-task={{GPUS_PER_TASK}}
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition={{PARTITION}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --output={{LOG_DIR}}/%x-%j.out
#SBATCH --error={{LOG_DIR}}/%x-%j.err
{{OPTIONAL_DIRECTIVES}}

{{COMMON}}
{{INPUT_CHECKS}}
exec {{VALIDATION_COMMAND}}
