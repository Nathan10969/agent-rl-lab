#!/usr/bin/env bash
# Pass the private cluster root, a new work directory, and the prior control directory.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s CLUSTER_ROOT WORK_DIR CONTROL_DIR\n' "$0" >&2
    exit 2
fi
cluster_root=$1
work_dir=$2
control_dir=$3
cd "$cluster_root"

headers_dir="$cluster_root/python-headers/include"
if [[ ! -r "$headers_dir/python3.10/Python.h" ||
      ! -r "$headers_dir/x86_64-linux-gnu/python3.10/pyconfig.h" ]]; then
    printf 'Python headers are missing from the private experiment directory.\n' >&2
    exit 1
fi
sha256sum --check official-shards.sha256

export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="$headers_dir/python3.10:$headers_dir${CPATH:+:$CPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$cluster_root/.alfworld-venv/bin/python" -u "$work_dir/project/a800/counted_hint_eval.py" \
    --manifest "$cluster_root/project-paired30-01/outputs/a800-paired-30-20261116/manifest.json" \
    --reference-dir "$cluster_root/results/paired30-01/alfworld-guide" \
    --control-dir "$control_dir" \
    --data-dir "$cluster_root/alfworld-data" \
    --output-dir "$work_dir/result"
