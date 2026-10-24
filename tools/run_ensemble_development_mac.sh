#!/bin/bash
# User-operated native handoff. Invoking this asserts an idle, AC-powered Mac.
# Scientific/readiness checks remain enforced by the published Python runner.
set -euo pipefail

elen_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elen_original="${ELEN_ORIGINAL_RUN:?Set ELEN_ORIGINAL_RUN to the original development results directory}"
elen_python="${ELEN_PYTHON:?Set ELEN_PYTHON to the project Python executable}"
elen_l1="${ELEN_L1_RUN:?Set ELEN_L1_RUN to the historical L1 results directory}"
elen_receipt="${ELEN_PUBLICATION_RECEIPT:?Set ELEN_PUBLICATION_RECEIPT to the verified publication receipt}"
elen_output="$elen_checkout/results/local_runs/l1-l2-independent-ensemble/development-completion"
elen_summary="$elen_checkout/results/local_runs/l1-l2-independent-ensemble/development-selection"
if [[ $# -gt 1 || ( $# -eq 1 && "${1-}" != '--retry-failed' ) ]]; then
  echo 'Usage: bash tools/run_ensemble_development_mac.sh [--retry-failed]' >&2
  exit 2
fi

cd "$elen_checkout"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f "$elen_output/plan.json" ]]; then
  "$elen_python" tools/run_cpu.py -m ensemble_study.development \
    --stage prepare --output-dir "$elen_output" \
    --original-dir "$elen_original" \
    --l1-dir "$elen_l1" --publication "$elen_receipt" \
    --operator user --host-ready
fi
"$elen_python" tools/run_cpu.py -m ensemble_study.development \
  --stage run --output-dir "$elen_output" --execute-development --resume \
  --operator user --host-ready "$@"
"$elen_python" tools/run_cpu.py -m ensemble_study.development \
  --stage summarize --output-dir "$elen_output" --summary-dir "$elen_summary" \
  --operator user --host-ready

echo "Development audit and selection saved to: $elen_summary"
echo 'Confirmation remains gated on publishing and verifying these selected settings.'
