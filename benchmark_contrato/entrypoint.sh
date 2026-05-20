#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
CONTRACT_ROOT="/app/benchmark_contrato"
RESULTS_DIR="${CONTRACT_ROOT}/results"
RAW_LOGS_DIR="${RESULTS_DIR}/raw_logs"

mkdir -p "${RAW_LOGS_DIR}"

if [[ -z "${SCENARIO}" ]]; then
  echo "Uso: ./benchmark_contrato/entrypoint.sh <official_reproduction|standardized_efficiency|regional_robustness>"
  exit 1
fi

python "${CONTRACT_ROOT}/export_results.py" preflight --scenario "${SCENARIO}" \
  > "${RAW_LOGS_DIR}/preflight_${SCENARIO}.log" 2>&1

case "${SCENARIO}" in
  official_reproduction)
    python "${CONTRACT_ROOT}/run_inference.py" --scenario official_reproduction \
      > "${RAW_LOGS_DIR}/run_inference_official.log" 2>&1
    ;;
  standardized_efficiency)
    python "${CONTRACT_ROOT}/profile.py" --scenario standardized_efficiency \
      > "${RAW_LOGS_DIR}/profile_efficiency.log" 2>&1
    ;;
  regional_robustness)
    python "${CONTRACT_ROOT}/run_inference.py" --scenario regional_robustness \
      > "${RAW_LOGS_DIR}/run_inference_regions.log" 2>&1
    ;;
  *)
    echo "Scenario invalido: ${SCENARIO}"
    exit 2
    ;;
esac

python "${CONTRACT_ROOT}/export_results.py" finalize --scenario "${SCENARIO}" \
  > "${RAW_LOGS_DIR}/finalize_${SCENARIO}.log" 2>&1

echo "Scenario ${SCENARIO} concluido. Resultados em ${RESULTS_DIR}."
