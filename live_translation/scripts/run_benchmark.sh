#!/usr/bin/env bash
# Run the full MuST-C benchmark on a CUDA device from the repository root.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

python_executable="${BENCHMARK_PYTHON:-python}"

# Show CLI help and validate paths without requiring an allocated GPU.
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" || "$argument" == "--check" ]]; then
    skip_cuda_check=1
    break
  fi
done

if [[ "${skip_cuda_check:-0}" != "1" ]]; then
  "$python_executable" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "A CUDA-enabled PyTorch installation and visible GPU are required for the full benchmark.")'
fi

for argument in "$@"; do
  if [[ "$argument" == "--config" || "$argument" == --config=* ]]; then
    exec "$python_executable" -m live_translation.benchmark --stage all "$@"
  fi
done

exec "$python_executable" -m live_translation.benchmark \
  --config live_translation/configs/full_ar.json --stage all "$@"
