#!/usr/bin/env bash
#
# Unattended 12-Run ViT Register Ablation Sweep
# =============================================
# Executes the full experimental matrix K in {0, 1, 4, 8} x seeds {42, 1337, 3407}
# sequentially on a single GPU, with zero manual intervention.
#
# Experiment matrix (global EXP index is fixed by the matrix position, so a
# subset run still produces canonically named directories):
#
#   EXP-01  K=0  seed 42     outputs/exp01_k0_s42/
#   EXP-02  K=0  seed 1337   outputs/exp02_k0_s1337/
#   EXP-03  K=0  seed 3407   outputs/exp03_k0_s3407/
#   EXP-04  K=1  seed 42     outputs/exp04_k1_s42/
#   EXP-05  K=1  seed 1337   outputs/exp05_k1_s1337/
#   EXP-06  K=1  seed 3407   outputs/exp06_k1_s3407/
#   EXP-07  K=4  seed 42     outputs/exp07_k4_s42/
#   EXP-08  K=4  seed 1337   outputs/exp08_k4_s1337/
#   EXP-09  K=4  seed 3407   outputs/exp09_k4_s3407/
#   EXP-10  K=8  seed 42     outputs/exp10_k8_s42/
#   EXP-11  K=8  seed 1337   outputs/exp11_k8_s1337/
#   EXP-12  K=8  seed 3407   outputs/exp12_k8_s3407/
#
# Operational guarantees:
#   * Failure resilience  - a crashing run is recorded in outputs/failures.log
#                           and the remaining runs continue.
#   * VRAM hygiene        - torch.cuda.empty_cache() is invoked between runs on
#                           top of the implicit release at process exit.
#   * Artifact contract   - every completed run is verified to contain
#                           metrics.json, best_model.pth, last_model.pth and
#                           train_history.csv.
#   * Checkpoint isolation- each run writes to checkpoints/<EXP_ID>/ so the three
#                           seeds of an arm never overwrite each other.
#   * Aggregation         - src/utils/logger.py is invoked at the end to emit
#                           outputs/sweep_summary.json.
#
# Usage:
#   bash scripts/run_sweep.sh                     # full 12-run matrix
#   bash scripts/run_sweep.sh --epochs 1          # fast end-to-end smoke sweep
#   bash scripts/run_sweep.sh --registers "0 4"   # subset of treatment arms
#   bash scripts/run_sweep.sh --seeds 3407        # subset of seeds
#   bash scripts/run_sweep.sh --plan              # print the plan, run nothing
#
# Exit status: 0 if every executed run succeeded, 1 if any run failed,
#              2 on an unknown option, 3 if the Python interpreter is missing.

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths: resolve the repository root from the script location so the sweep can
# be launched from any working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_CONFIG="configs/sweep_config.yaml"
OUTPUT_ROOT="outputs"
CHECKPOINT_ROOT="checkpoints"
DATA_DIR=""
EPOCHS=""
EXTRA_ARGS=""
SKIP_EXISTING=0
RUN_AGGREGATOR=1
PLAN_ONLY=0

DEFAULT_REGISTERS="0 1 4 8"
DEFAULT_SEEDS="42 1337 3407"
REGISTERS=""
SEEDS=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_sweep.sh [options]

Options:
  -e, --epochs N            Override training epochs (smoke tests use 1)
  -r, --registers "0 4"     Space separated subset of register arms to run
  -s, --seeds "42 1337"     Space separated subset of seeds to run
  -d, --data-dir PATH       Dataset root passed to scripts/train.py
  -o, --output-root PATH    Root for per-run output directories (default: outputs)
  -c, --checkpoint-root PATH
                            Root for per-run checkpoint directories (default: checkpoints)
  -p, --python BIN          Python interpreter to use (default: python)
      --extra-args "..."    Additional raw arguments forwarded to scripts/train.py
      --skip-existing       Skip runs whose metrics.json already exists (resume)
      --no-aggregate        Do not run src/utils/logger.py at the end
      --plan                Print the execution plan without training
                            (the Definition-of-Done smoke sweep is --epochs 1,
                             which does train; --plan trains nothing at all)
  -h, --help                Show this message and exit
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--epochs)           EPOCHS="$2"; shift 2 ;;
    -r|--registers)        REGISTERS="$2"; shift 2 ;;
    -s|--seeds)            SEEDS="$2"; shift 2 ;;
    -d|--data-dir)         DATA_DIR="$2"; shift 2 ;;
    -o|--output-root)      OUTPUT_ROOT="$2"; shift 2 ;;
    -c|--checkpoint-root)  CHECKPOINT_ROOT="$2"; shift 2 ;;
    -p|--python)           PYTHON_BIN="$2"; shift 2 ;;
    --extra-args)          EXTRA_ARGS="$2"; shift 2 ;;
    --skip-existing)       SKIP_EXISTING=1; shift ;;
    --no-aggregate)        RUN_AGGREGATOR=0; shift ;;
    --plan|--dry-run)      PLAN_ONLY=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Interpreter resolution. A missing interpreter would otherwise fail all twelve
# runs identically, so it is resolved once, up front, and the sweep refuses to
# start rather than filling failures.log with the same error twelve times.
# ---------------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Interpreter '${PYTHON_BIN}' not found on PATH. Activate the environment or pass --python." >&2
    exit 3
  fi
fi

# ---------------------------------------------------------------------------
# Matrix definition. The register arms and seeds are read from
# configs/sweep_config.yaml so the runner and the documented sweep definition
# can never drift apart; the hard-coded matrix is used as a fallback.
# ---------------------------------------------------------------------------
read_sweep_config() {
  # $1: dotted parameter name inside configs/sweep_config.yaml
  "${PYTHON_BIN}" - "$1" <<'PY' 2>/dev/null
import sys
import yaml

param = sys.argv[1]
with open("configs/sweep_config.yaml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
values = config.get("parameters", {}).get(param, {}).get("values", [])
if not values:
    raise SystemExit(1)
print(" ".join(str(v) for v in values))
PY
}

MATRIX_REGISTERS=""
MATRIX_SEEDS=""
if [[ -f "${SWEEP_CONFIG}" ]]; then
  MATRIX_REGISTERS="$(read_sweep_config num_registers)"
  MATRIX_SEEDS="$(read_sweep_config seed)"
fi
[[ -z "${MATRIX_REGISTERS}" ]] && MATRIX_REGISTERS="${DEFAULT_REGISTERS}"
[[ -z "${MATRIX_SEEDS}" ]] && MATRIX_SEEDS="${DEFAULT_SEEDS}"

read -r -a MATRIX_REGISTER_ARRAY <<< "${MATRIX_REGISTERS}"
read -r -a MATRIX_SEED_ARRAY <<< "${MATRIX_SEEDS}"

# Selection defaults to the full matrix.
[[ -z "${REGISTERS}" ]] && REGISTERS="${MATRIX_REGISTERS}"
[[ -z "${SEEDS}" ]] && SEEDS="${MATRIX_SEEDS}"
read -r -a SELECTED_REGISTERS <<< "${REGISTERS}"
read -r -a SELECTED_SEEDS <<< "${SEEDS}"

# Maps a register count to its treatment arm configuration file.
config_for_k() {
  case "$1" in
    0) echo "configs/baseline_k0.yaml" ;;
    1) echo "configs/vit_tiny_k1.yaml" ;;
    4) echo "configs/vit_tiny_k4.yaml" ;;
    8) echo "configs/vit_tiny_k8.yaml" ;;
    *) echo "" ;;
  esac
}

contains() {
  # $1: needle, $2..: haystack
  local needle="$1"; shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Logging targets
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_ROOT}" "${CHECKPOINT_ROOT}"
SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
FAILURE_LOG="${OUTPUT_ROOT}/failures.log"
: > "${SWEEP_LOG}"

log() {
  # Mirrors a line to the console and to the cumulative sweep log.
  echo "$@" | tee -a "${SWEEP_LOG}"
}

REQUIRED_ARTIFACTS=("metrics.json" "best_model.pth" "last_model.pth" "train_history.csv")

verify_artifacts() {
  # $1: run directory. Echoes the names of any missing contract artifacts.
  local run_dir="$1"
  local missing=()
  local artifact
  for artifact in "${REQUIRED_ARTIFACTS[@]}"; do
    [[ -f "${run_dir}/${artifact}" ]] || missing+=("${artifact}")
  done
  echo "${missing[*]:-}"
}

clear_gpu_memory() {
  # Sequential subprocesses release VRAM at exit; the explicit cache flush
  # guarantees the allocator is defragmented before the next run starts.
  "${PYTHON_BIN}" -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" \
    >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
TOTAL_PLANNED=$(( ${#SELECTED_REGISTERS[@]} * ${#SELECTED_SEEDS[@]} ))
SWEEP_START=$(date +%s)

log "=========================================================="
log "Starting ${TOTAL_PLANNED}-Run ViT Register Ablation Sweep Matrix"
log "=========================================================="
log "Repository      : ${REPO_ROOT}"
log "Interpreter     : ${PYTHON_BIN}"
log "Register arms   : ${REGISTERS}"
log "Seeds           : ${SEEDS}"
log "Output root     : ${OUTPUT_ROOT}"
log "Checkpoint root : ${CHECKPOINT_ROOT}"
[[ -n "${EPOCHS}" ]] && log "Epoch override  : ${EPOCHS}"
[[ ${PLAN_ONLY} -eq 1 ]] && log "Mode            : PLAN ONLY (nothing is trained)"
log "Started         : $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------
EXP_IDX=0
RUN_COUNTER=0
SUCCESS_COUNT=0
FAILURE_COUNT=0
SKIPPED_COUNT=0
FAILED_EXPERIMENTS=()

for K in "${MATRIX_REGISTER_ARRAY[@]}"; do
  CONFIG="$(config_for_k "${K}")"

  for SEED in "${MATRIX_SEED_ARRAY[@]}"; do
    # The global experiment index advances over the whole matrix, so directory
    # names stay canonical even when only a subset is selected.
    EXP_IDX=$((EXP_IDX + 1))

    contains "${K}" "${SELECTED_REGISTERS[@]}" || continue
    contains "${SEED}" "${SELECTED_SEEDS[@]}" || continue

    EXP_ID="$(printf "exp%02d_k%d_s%d" "${EXP_IDX}" "${K}" "${SEED}")"
    RUN_DIR="${OUTPUT_ROOT}/${EXP_ID}"
    CKPT_DIR="${CHECKPOINT_ROOT}/${EXP_ID}"
    RUN_LOG="${OUTPUT_ROOT}/${EXP_ID}.log"
    RUN_COUNTER=$((RUN_COUNTER + 1))

    if [[ -z "${CONFIG}" ]]; then
      log ""
      log "!! [${RUN_COUNTER}/${TOTAL_PLANNED}] No configuration file registered for K=${K}; skipping ${EXP_ID}."
      echo "$(date '+%Y-%m-%d %H:%M:%S') | ${EXP_ID} | no config file registered for K=${K}" >> "${FAILURE_LOG}"
      FAILURE_COUNT=$((FAILURE_COUNT + 1))
      FAILED_EXPERIMENTS+=("${EXP_ID}")
      continue
    fi

    if [[ ${SKIP_EXISTING} -eq 1 && -f "${RUN_DIR}/metrics.json" ]]; then
      log ""
      log ">> [${RUN_COUNTER}/${TOTAL_PLANNED}] ${EXP_ID} already complete; skipping (--skip-existing)."
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
      continue
    fi

    mkdir -p "${RUN_DIR}" "${CKPT_DIR}"

    TRAIN_ARGS=(
      --config "${CONFIG}"
      --num_registers "${K}"
      --seed "${SEED}"
      --output_dir "${RUN_DIR}"
      --checkpoint_dir "${CKPT_DIR}"
    )
    [[ -n "${EPOCHS}" ]] && TRAIN_ARGS+=(--epochs "${EPOCHS}")
    [[ -n "${DATA_DIR}" ]] && TRAIN_ARGS+=(--data_dir "${DATA_DIR}")
    if [[ -n "${EXTRA_ARGS}" ]]; then
      read -r -a EXTRA_ARG_ARRAY <<< "${EXTRA_ARGS}"
      TRAIN_ARGS+=("${EXTRA_ARG_ARRAY[@]}")
    fi

    log ""
    log "----------------------------------------------------------"
    log ">> [${RUN_COUNTER}/${TOTAL_PLANNED}] Executing ${EXP_ID} (K=${K}, Seed=${SEED})"
    log "   config=${CONFIG}  outputs=${RUN_DIR}  checkpoints=${CKPT_DIR}"
    log "----------------------------------------------------------"

    if [[ ${PLAN_ONLY} -eq 1 ]]; then
      log "   [plan] ${PYTHON_BIN} scripts/train.py ${TRAIN_ARGS[*]}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      continue
    fi

    RUN_START=$(date +%s)
    "${PYTHON_BIN}" scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_LOG}" | tee -a "${SWEEP_LOG}"
    STATUS=${PIPESTATUS[0]}
    RUN_ELAPSED=$(( $(date +%s) - RUN_START ))

    if [[ ${STATUS} -ne 0 ]]; then
      # Failure resilience: record and continue with the remaining matrix.
      FAILURE_COUNT=$((FAILURE_COUNT + 1))
      FAILED_EXPERIMENTS+=("${EXP_ID}")
      echo "$(date '+%Y-%m-%d %H:%M:%S') | ${EXP_ID} | K=${K} seed=${SEED} | exit=${STATUS} | log=${RUN_LOG}" >> "${FAILURE_LOG}"
      log "   FAILED after ${RUN_ELAPSED}s with exit code ${STATUS} (recorded in ${FAILURE_LOG}); continuing."
    else
      MISSING="$(verify_artifacts "${RUN_DIR}")"
      if [[ -n "${MISSING}" ]]; then
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        FAILED_EXPERIMENTS+=("${EXP_ID}")
        echo "$(date '+%Y-%m-%d %H:%M:%S') | ${EXP_ID} | K=${K} seed=${SEED} | incomplete artifacts: ${MISSING}" >> "${FAILURE_LOG}"
        log "   COMPLETED in ${RUN_ELAPSED}s but artifacts are incomplete: ${MISSING}"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log "   Completed in ${RUN_ELAPSED}s; all contract artifacts present."
      fi
    fi

    clear_gpu_memory
  done
done

SWEEP_ELAPSED=$(( $(date +%s) - SWEEP_START ))

# ---------------------------------------------------------------------------
# Aggregation and final report
# ---------------------------------------------------------------------------
log ""
log "=========================================================="
log "Sweep finished in $((SWEEP_ELAPSED / 60))m $((SWEEP_ELAPSED % 60))s"
log "Succeeded: ${SUCCESS_COUNT} | Failed: ${FAILURE_COUNT} | Skipped: ${SKIPPED_COUNT} | Planned: ${TOTAL_PLANNED}"
if [[ ${FAILURE_COUNT} -gt 0 ]]; then
  log "Failed experiments: ${FAILED_EXPERIMENTS[*]}"
  log "See ${FAILURE_LOG} for details."
fi
log "=========================================================="

if [[ ${RUN_AGGREGATOR} -eq 1 && ${PLAN_ONLY} -eq 0 ]]; then
  log "Aggregating metrics across seeds..."
  "${PYTHON_BIN}" src/utils/logger.py --output_dir "${OUTPUT_ROOT}" 2>&1 | tee -a "${SWEEP_LOG}"
  log "=========================================================="
fi

[[ ${FAILURE_COUNT} -eq 0 ]] && exit 0 || exit 1
