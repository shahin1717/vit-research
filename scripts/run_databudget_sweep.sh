#!/usr/bin/env bash
#
# Unattended 12-Run ViT Register Ablation Sweep (300 Images/Class Data Budget Extension)
# ======================================================================================
# Executes the 300 images/class extension matrix: K in {0, 1, 4, 8} x seeds {42, 1337, 3407}
# sequentially on a single GPU.
#
# Design Principle:
#   Strictly additive. Does NOT overwrite outputs/, checkpoints/, or sweep_summary.json.
#   All artifacts are routed to outputs_databudget/ and checkpoints_databudget/.
#
# Experiment matrix (runs 13 through 24):
#   EXP-13  K=0  seed 42     outputs_databudget/exp13_k0_pc300_s42/
#   EXP-14  K=0  seed 1337   outputs_databudget/exp14_k0_pc300_s1337/
#   EXP-15  K=0  seed 3407   outputs_databudget/exp15_k0_pc300_s3407/
#   EXP-16  K=1  seed 42     outputs_databudget/exp16_k1_pc300_s42/
#   EXP-17  K=1  seed 1337   outputs_databudget/exp17_k1_pc300_s1337/
#   EXP-18  K=1  seed 3407   outputs_databudget/exp18_k1_pc300_s3407/
#   EXP-19  K=4  seed 42     outputs_databudget/exp19_k4_pc300_s42/
#   EXP-20  K=4  seed 1337   outputs_databudget/exp20_k4_pc300_s1337/
#   EXP-21  K=4  seed 3407   outputs_databudget/exp21_k4_pc300_s3407/
#   EXP-22  K=8  seed 42     outputs_databudget/exp22_k8_pc300_s42/
#   EXP-23  K=8  seed 1337   outputs_databudget/exp23_k8_pc300_s1337/
#   EXP-24  K=8  seed 3407   outputs_databudget/exp24_k8_pc300_s3407/
#
# Usage:
#   bash scripts/run_databudget_sweep.sh                     # full 12-run sweep (~8.5 GPU-hours)
#   bash scripts/run_databudget_sweep.sh --epochs 1          # smoke test across all arms
#   bash scripts/run_databudget_sweep.sh -r "0" -s "42" -e 1  # 1-arm fast test
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# Detect python binary: prefer mlaiac conda env if available
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/shahin/miniconda3/envs/mlaiac/bin/python" ]]; then
    PYTHON_BIN="/home/shahin/miniconda3/envs/mlaiac/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

OUTPUT_ROOT="outputs_databudget"
CHECKPOINT_ROOT="checkpoints_databudget"
EPOCHS=""
BUDGET="300"
REGISTERS="0 1 4 8"
SEEDS="42 1337 3407"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--epochs)    EPOCHS="$2"; shift 2 ;;
    -r|--registers) REGISTERS="$2"; shift 2 ;;
    -s|--seeds)     SEEDS="$2"; shift 2 ;;
    -p|--python)    PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: run_databudget_sweep.sh [-e epochs] [-r \"0 1 4 8\"] [-s \"42 1337 3407\"] [-p python_path]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}" "${CHECKPOINT_ROOT}"
SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
FAILURE_LOG="${OUTPUT_ROOT}/failures.log"
touch "${SWEEP_LOG}"

config_prefix_for_k() {
  case "$1" in
    0) echo "configs/baseline_k0" ;;
    1) echo "configs/vit_tiny_k1" ;;
    4) echo "configs/vit_tiny_k4" ;;
    8) echo "configs/vit_tiny_k8" ;;
    *) echo "" ;;
  esac
}

echo "====================================================================" | tee -a "${SWEEP_LOG}"
echo "🚀 Starting 300 Images/Class Data Budget Extension Sweep" | tee -a "${SWEEP_LOG}"
echo "Interpreter: ${PYTHON_BIN}" | tee -a "${SWEEP_LOG}"
echo "Arms: K in [${REGISTERS}] | Seeds: [${SEEDS}]" | tee -a "${SWEEP_LOG}"
echo "====================================================================" | tee -a "${SWEEP_LOG}"

EXP_IDX=12   # Continue numbering after original exp01..exp12
ANY_FAILED=0

for K in ${REGISTERS}; do
  PREFIX="$(config_prefix_for_k "${K}")"
  CONFIG="${PREFIX}_${BUDGET}pc.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "!! Missing config ${CONFIG}, skipping K=${K}" | tee -a "${SWEEP_LOG}"
    continue
  fi
  for SEED in ${SEEDS}; do
    EXP_IDX=$((EXP_IDX + 1))
    EXP_ID="$(printf "exp%02d_k%d_pc%d_s%d" "${EXP_IDX}" "${K}" "${BUDGET}" "${SEED}")"
    RUN_DIR="${OUTPUT_ROOT}/${EXP_ID}"
    CKPT_DIR="${CHECKPOINT_ROOT}/${EXP_ID}"
    mkdir -p "${RUN_DIR}" "${CKPT_DIR}"

    TRAIN_ARGS=(--config "${CONFIG}" --k_registers "${K}" --seed "${SEED}" \
                --output_dir "${RUN_DIR}" --checkpoint_dir "${CKPT_DIR}")
    [[ -n "${EPOCHS}" ]] && TRAIN_ARGS+=(--epochs "${EPOCHS}")

    echo "" | tee -a "${SWEEP_LOG}"
    echo ">> ${EXP_ID}  (K=${K}, budget=${BUDGET}/class, seed=${SEED})" | tee -a "${SWEEP_LOG}"

    RUN_START=$(date +%s)
    "${PYTHON_BIN}" scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}.log" | tee -a "${SWEEP_LOG}"
    STATUS=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - RUN_START ))

    if [[ ${STATUS} -ne 0 ]]; then
      ANY_FAILED=1
      echo "$(date '+%F %T') | ${EXP_ID} | exit=${STATUS}" >> "${FAILURE_LOG}"
      echo "   ❌ FAILED after ${ELAPSED}s (exit ${STATUS})" | tee -a "${SWEEP_LOG}"
    else
      echo "   ✅ Completed in ${ELAPSED}s" | tee -a "${SWEEP_LOG}"
    fi

    "${PYTHON_BIN}" -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" \
      >/dev/null 2>&1 || true
  done
done

echo "" | tee -a "${SWEEP_LOG}"
echo "====================================================================" | tee -a "${SWEEP_LOG}"
echo "300pc sweep completed. Aggregate results with:" | tee -a "${SWEEP_LOG}"
echo "  ${PYTHON_BIN} src/utils/aggregate_databudget.py --output_dir ${OUTPUT_ROOT}" | tee -a "${SWEEP_LOG}"
echo "====================================================================" | tee -a "${SWEEP_LOG}"

exit ${ANY_FAILED}
