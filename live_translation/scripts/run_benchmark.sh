#!/usr/bin/env bash
# Run the full MuST-C benchmark from the repository root.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

python_executable="${BENCHMARK_PYTHON:-python}"
for argument in "$@"; do
  if [[ "$argument" == "--config" || "$argument" == --config=* ]]; then
    exec "$python_executable" -m live_translation.benchmark "$@"
  fi
done

exec "$python_executable" -m live_translation.benchmark \
  --config live_translation/configs/full_ar.json "$@"
