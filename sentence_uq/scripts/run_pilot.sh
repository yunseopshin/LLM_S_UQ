#!/usr/bin/env bash
# scripts/run_pilot.sh — End-to-end pilot pipeline for Bayesian sentence-level
# factuality UQ (Phase 7-1 integration).
#
# Usage
# -----
#   bash scripts/run_pilot.sh                         # resume completed phases
#   bash scripts/run_pilot.sh --force                 # ignore stamps, redo all
#   bash scripts/run_pilot.sh --config configs/pilot.yaml
#   bash scripts/run_pilot.sh --label smoke           # use results/smoke if no
#                                                     # results_dir in config
#
# Behaviour
# ---------
#   * Each phase runs only if its stamp file is missing (`--force` overrides).
#   * Per-phase output is duplicated to a timestamped log under
#     ${RESULTS_DIR}/logs/<phase>_<ts>.log.
#   * On failure the script prints the failing phase, exit code, and log
#     location, then exits with the same code. Stamps are only written when a
#     phase finishes successfully — re-running resumes from the failed step.
#   * A consolidated summary is appended to
#     ${RESULTS_DIR}/logs/summary_<ts>.txt and printed at the end.
#   * Phase 4 (auxiliary head) is opportunistic: it runs only when both the
#     trained main model and an offline U^* file are present; otherwise it is
#     reported as skipped.

set -u
set -o pipefail

# --- project root ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- argument parsing --------------------------------------------------------
CONFIG="configs/pilot.yaml"
RESULTS_DIR=""
LABEL="pilot"
FORCE=0

usage() {
    cat <<USAGE
Usage: $0 [--config CFG] [--results-dir DIR] [--label NAME] [--force]
  --config CFG         YAML config (default: configs/pilot.yaml)
  --results-dir DIR    Override results dir (default: cfg.results_dir or results/<label>)
  --label NAME         Fallback subdir under results/ (default: pilot)
  --force              Re-run every phase regardless of stamp files
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";        shift 2 ;;
        --config=*)     CONFIG="${1#*=}";   shift ;;
        --results-dir)  RESULTS_DIR="$2";   shift 2 ;;
        --results-dir=*) RESULTS_DIR="${1#*=}"; shift ;;
        --label)        LABEL="$2";         shift 2 ;;
        --label=*)      LABEL="${1#*=}";    shift ;;
        --force)        FORCE=1;            shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "[run_pilot] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "${CONFIG}" ]]; then
    echo "[run_pilot] config not found: ${CONFIG}" >&2
    exit 2
fi

# Derive results dir from config (results_dir key) if not provided on the CLI.
if [[ -z "${RESULTS_DIR}" ]]; then
    RESULTS_DIR="$(python - "$CONFIG" "$LABEL" <<'PY'
import sys
try:
    import yaml
except ImportError:
    print(f"results/{sys.argv[2]}")
    sys.exit(0)
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("results_dir") or f"results/{sys.argv[2]}")
PY
)"
fi

LOG_DIR="${RESULTS_DIR}/logs"
STAMP_DIR="${RESULTS_DIR}/stamps"
mkdir -p "${LOG_DIR}" "${STAMP_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/summary_${TIMESTAMP}.txt"
PIPELINE_START=$SECONDS

# --- helpers -----------------------------------------------------------------
log_summary() { printf '%s\n' "$*" | tee -a "${SUMMARY}" ; }

run_phase() {
    # run_phase <phase_id> <description> -- <command...>
    local phase="$1"; shift
    local desc="$1";  shift
    if [[ "${1:-}" == "--" ]]; then shift; fi

    local stamp="${STAMP_DIR}/${phase}.done"
    local log="${LOG_DIR}/${phase}_${TIMESTAMP}.log"

    if [[ "${FORCE}" -eq 0 && -f "${stamp}" ]]; then
        log_summary "[SKIP] ${phase}: ${desc}  (stamp present: ${stamp})"
        return 0
    fi

    echo
    echo "=== ${phase}: ${desc} ==="
    echo "    cmd : $*"
    echo "    log : ${log}"
    local start=$SECONDS

    set +e
    ( "$@" ) 2>&1 | tee "${log}"
    local rc="${PIPESTATUS[0]}"
    set -e

    local dur=$((SECONDS - start))
    if [[ "${rc}" -ne 0 ]]; then
        log_summary "[FAIL] ${phase}: ${desc}  (exit ${rc}, ${dur}s)"
        echo                                                                >&2
        echo "ERROR: phase '${phase}' failed with exit code ${rc}."         >&2
        echo "       log : ${log}"                                          >&2
        echo "       stamp NOT written; re-running this script resumes"     >&2
        echo "       from this phase (use --force to redo earlier ones)."   >&2
        exit "${rc}"
    fi
    touch "${stamp}"
    log_summary "[OK]   ${phase}: ${desc}  (${dur}s)"
}

# --- banner ------------------------------------------------------------------
log_summary "=================================================================="
log_summary "Bayesian sentence-level UQ pipeline (label=${LABEL})"
log_summary "Config        : ${CONFIG}"
log_summary "Results dir   : ${RESULTS_DIR}"
log_summary "Log dir       : ${LOG_DIR}"
log_summary "Stamp dir     : ${STAMP_DIR}"
log_summary "Timestamp     : ${TIMESTAMP}"
log_summary "Force re-run  : ${FORCE}"
log_summary "Project root  : ${PROJECT_ROOT}"
log_summary "=================================================================="

# --- pipeline ----------------------------------------------------------------
run_phase phase0  "Prepare dataset (raw + split)" -- \
    python scripts/00_prepare_dataset.py --config "${CONFIG}"

run_phase phase1  "Generate hidden states + token mapping" -- \
    python scripts/01_generate_data.py --config "${CONFIG}"

run_phase phase1b "Cache per-token entropy / top-1" -- \
    python scripts/01b_cache_scalars.py --config "${CONFIG}"

run_phase phase2  "Annotate factuality (K_j, m_j)" -- \
    python scripts/02_annotate_factuality.py --config "${CONFIG}"

run_phase phase3  "Train main Bayesian model" -- \
    python scripts/03_train.py --config "${CONFIG}"

# Phase 4 (auxiliary regression head) needs Phase 3's checkpoint plus an
# offline-computed U^* file. Skip cleanly if either is missing.
TRAINED_MODEL="${RESULTS_DIR}/trained_model.pt"
USTAR_FILE=""
for candidate in \
    "data/processed/u_star.json" \
    "data/processed/u_star.pt" \
    "data/processed/u_star_setup_1.json" \
    "data/processed/u_star_setup_1.pt" \
    "data/processed/u_star_setup_2.json" \
    "data/processed/u_star_setup_2.pt" \
    "data/processed/u_star_setup_3.json" \
    "data/processed/u_star_setup_3.pt" ; do
    if [[ -f "${candidate}" ]]; then USTAR_FILE="${candidate}"; break; fi
done

if [[ -f "${TRAINED_MODEL}" && -n "${USTAR_FILE}" ]]; then
    run_phase phase4 "Train auxiliary Bayesian head" -- \
        python scripts/04_train_aux.py --config "${CONFIG}" \
            --trained-model "${TRAINED_MODEL}" \
            --u-star "${USTAR_FILE}"
else
    reason="trained model: ${TRAINED_MODEL}"
    [[ -f "${TRAINED_MODEL}" ]] && reason="${reason} OK"  || reason="${reason} MISSING"
    if [[ -n "${USTAR_FILE}" ]]; then
        reason="${reason}; U^*: ${USTAR_FILE}"
    else
        reason="${reason}; U^*: not found under data/processed/"
    fi
    log_summary "[SKIP] phase4: aux head  (${reason})"
fi

run_phase phase5  "Baselines (incl. Han et al. factuality probe)" -- \
    python scripts/05_baselines.py --config "${CONFIG}"

run_phase phase6  "Evaluate (ratio-level + strict + comparisons)" -- \
    python scripts/04_evaluate.py --config "${CONFIG}"

# --- summary -----------------------------------------------------------------
TOTAL=$((SECONDS - PIPELINE_START))
log_summary "------------------------------------------------------------------"
log_summary "Pipeline finished in ${TOTAL}s"
log_summary "Results : ${RESULTS_DIR}"
log_summary "Summary : ${SUMMARY}"
log_summary "------------------------------------------------------------------"

cat <<'CHECKLIST'

Experiment checklist (verify against results/):
  [ ] 10-entity smoke test passed
  [ ] 50-entity pilot complete, all metrics computed
  [ ] Ratio-level: MAE and Pearson r reasonable (primary evaluation)
  [ ] Strict AUROC at least comparable to baselines
  [ ] Bayesian ECE < Point estimate ECE (core hypothesis)
  [ ] Our ECE < Factuality Probe (Han et al.) ECE (key comparison)
  [ ] Binomial NLL computed and reasonable (count-aware model advantage)
  [ ] Rejection curve: Ours comparable or better than Han et al.
  [ ] MC vs linear correlation > 0.9
  [ ] Learned alpha distribution visualized (compare with Han et al. layer 14)
  [ ] m_j distribution checked (no excessive m_j=0, no extreme dominance)

CHECKLIST

echo "Done. Results in ${RESULTS_DIR}"
