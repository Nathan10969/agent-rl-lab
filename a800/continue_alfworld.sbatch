#!/usr/bin/env bash
# Pass the existing private cluster root and a new private work directory.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s CLUSTER_ROOT WORK_DIR\n' "$0" >&2
    exit 2
fi
cluster_root=$1
work_dir=$2
cd "$cluster_root"

headers_dir="$cluster_root/python-headers/include"
if [[ ! -r "$headers_dir/python3.10/Python.h" ||
      ! -r "$headers_dir/x86_64-linux-gnu/python3.10/pyconfig.h" ]]; then
    printf 'Python headers are missing from the private experiment directory.\n' >&2
    exit 1
fi

# The original run recorded these hashes for the official FP8 weight shards.
sha256sum --check official-shards.sha256

# Match the working baseline: avoid FlashInfer's nvcc-dependent sampler and
# provide the same private Python development headers for other compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0
export CPATH="$headers_dir/python3.10:$headers_dir${CPATH:+:$CPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$cluster_root/.alfworld-venv/bin/python" -u "$work_dir/project/a800/continue_alfworld.py" \
    --manifest "$cluster_root/project-paired30-01/outputs/a800-paired-30-20261116/manifest.json" \
    --reference-dir "$cluster_root/results/paired30-01/alfworld-guide" \
    --data-dir "$cluster_root/alfworld-data" \
    --output-dir "$work_dir/result"
