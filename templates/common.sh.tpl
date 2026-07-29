set -euo pipefail
umask 077
mkdir -p "{{LOG_DIR}}"
cd "{{WORKSPACE}}"
{{MODULE_LOADS}}
{{CONDA_INIT}}
{{CONDA_ACTIVATE}}
