#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs checkpoints data/cache worldgen exports
LOG_FILE="logs/colab_pipeline_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Starting unattended Colab pipeline"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Log file: ${LOG_FILE}"

trap 'echo "[ERROR] Pipeline failed at line $LINENO. See ${LOG_FILE}"' ERR

timestamp() {
	date +"%Y-%m-%d %H:%M:%S"
}

log() {
	echo "[$(timestamp)] $*"
}

run_step() {
	local step_name="$1"
	shift
	log "===== ${step_name} ====="
	"$@"
	log "===== Completed: ${step_name} ====="
}

retry() {
	local attempts="$1"
	shift
	local delay_seconds="$1"
	shift

	local n=1
	until "$@"; do
		if [[ "$n" -ge "$attempts" ]]; then
			log "Command failed after ${attempts} attempts: $*"
			return 1
		fi
		log "Attempt ${n}/${attempts} failed. Retrying in ${delay_seconds}s: $*"
		sleep "${delay_seconds}"
		n=$((n + 1))
	done
}

# -----------------------------
# Config (override via env vars)
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-${PYTHON_BIN} -m pip}"

APT_PACKAGES="${APT_PACKAGES:-curl unzip git ca-certificates}"
WORLDGEN_RETRIES="${WORLDGEN_RETRIES:-2}"

NUM_WORLDS="${NUM_WORLDS:-20}"
WORLDGEN_PARALLEL="${WORLDGEN_PARALLEL:-$(nproc 2>/dev/null || echo 2)}"

RUN_ANALYZE="${RUN_ANALYZE:-0}"
ANALYZE_SOURCE_DIR="${ANALYZE_SOURCE_DIR:-worldgen}"
ANALYZE_OUTPUT="${ANALYZE_OUTPUT:-src/terraria/natural_ids.py}"

PREP_SOURCE="${PREP_SOURCE:-worldgen}"
PREP_CHUNK_SIZE="${PREP_CHUNK_SIZE:-32}"
PREP_OVERLAP="${PREP_OVERLAP:-8}"
PREP_MIN_DIVERSITY="${PREP_MIN_DIVERSITY:-0.20}"
PREP_OUTPUT_DIR="${PREP_OUTPUT_DIR:-data/cache}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_DISK_MODE="${TRAIN_DISK_MODE:-1}"
TRAIN_CACHE_SIZE="${TRAIN_CACHE_SIZE:-5}"
TRAIN_DATA="${TRAIN_DATA:-${PREP_OUTPUT_DIR}}"
TRAIN_EMA_DECAY="${TRAIN_EMA_DECAY:-0.99}"
TRAIN_EMA_RESET_MULTIPLIER="${TRAIN_EMA_RESET_MULTIPLIER:-0.5}"
TRAIN_EMA_RESET_INTERVAL="${TRAIN_EMA_RESET_INTERVAL:-500}"
TRAIN_BETA="${TRAIN_BETA:-0.25}"
TRAIN_METRICS_STRIDE="${TRAIN_METRICS_STRIDE:-50}"

USE_BLOCK_LOSS_WEIGHTED="${USE_BLOCK_LOSS_WEIGHTED:-0}"
BLOCK_WEIGHT_MIN="${BLOCK_WEIGHT_MIN:-0.5}"
BLOCK_WEIGHT_MAX="${BLOCK_WEIGHT_MAX:-5.0}"

RESUME_MODE="${RESUME_MODE:-auto}" # auto|always|never
RESUME_PATH="${RESUME_PATH:-checkpoints/latest_model.pt}"

CLI_MODE="${CLI_MODE:-auto}" # auto|terrain|module

install_system_deps() {
	export DEBIAN_FRONTEND=noninteractive
	if command -v apt-get >/dev/null 2>&1; then
		retry 3 10 apt-get update
		retry 3 10 apt-get install -y ${APT_PACKAGES}
	else
		log "apt-get not found; skipping system package installation"
	fi
}

install_python_deps() {
	${PYTHON_BIN} --version

	retry 3 8 ${PIP_BIN} install --upgrade pip setuptools wheel

	# Try editable install first (enables `terrain` entrypoint).
	if retry 2 8 ${PIP_BIN} install -e .; then
		log "Editable install completed"
	else
		log "Editable install failed; installing dependencies directly as fallback"
		retry 3 10 ${PIP_BIN} install \
			lihzahrd \
			matplotlib \
			numpy \
			rich \
			typer \
			tqdm \
			torch \
			"terraschem @ git+https://github.com/Neels-v-Wyk/terraschem.git"
	fi
}

detect_cli() {
	if [[ "${CLI_MODE}" == "terrain" ]]; then
		TERRAIN_CMD=(terrain)
		return
	fi

	if [[ "${CLI_MODE}" == "module" ]]; then
		TERRAIN_CMD=("${PYTHON_BIN}" -m src.cli)
		return
	fi

	if command -v terrain >/dev/null 2>&1; then
		TERRAIN_CMD=(terrain)
	else
		TERRAIN_CMD=("${PYTHON_BIN}" -m src.cli)
	fi
}

build_train_args() {
	TRAIN_ARGS=(
		model train
		--data "${TRAIN_DATA}"
		--epochs "${TRAIN_EPOCHS}"
		--batch-size "${TRAIN_BATCH_SIZE}"
		--cache-size "${TRAIN_CACHE_SIZE}"
		--ema-decay "${TRAIN_EMA_DECAY}"
		--ema-reset-multiplier "${TRAIN_EMA_RESET_MULTIPLIER}"
		--ema-reset-interval "${TRAIN_EMA_RESET_INTERVAL}"
		--beta "${TRAIN_BETA}"
		--block-weight-min "${BLOCK_WEIGHT_MIN}"
		--block-weight-max "${BLOCK_WEIGHT_MAX}"
		--metrics-stride "${TRAIN_METRICS_STRIDE}"
	)

	if [[ "${TRAIN_DISK_MODE}" == "1" ]]; then
		TRAIN_ARGS+=(--disk-mode)
	fi

	if [[ "${USE_BLOCK_LOSS_WEIGHTED}" == "1" ]]; then
		TRAIN_ARGS+=(--block-loss-weighted)
	fi

	case "${RESUME_MODE}" in
		auto)
			if [[ -f "${RESUME_PATH}" ]]; then
				TRAIN_ARGS+=(--resume "${RESUME_PATH}")
				log "Auto-resume enabled: ${RESUME_PATH}"
			else
				log "Auto-resume skipped; checkpoint not found at ${RESUME_PATH}"
			fi
			;;
		always)
			TRAIN_ARGS+=(--resume "${RESUME_PATH}")
			;;
		never)
			;;
		*)
			log "Invalid RESUME_MODE='${RESUME_MODE}' (expected auto|always|never)"
			return 1
			;;
	esac
}

run_pipeline() {
	chmod +x src/scripts/worldgen.sh || true

	run_step "Install system dependencies" install_system_deps
	run_step "Install Python dependencies" install_python_deps

	detect_cli
	log "Using CLI command: ${TERRAIN_CMD[*]}"

	run_step "Generate worlds" retry "${WORLDGEN_RETRIES}" 20 \
		"${TERRAIN_CMD[@]}" data worldgen --num-worlds "${NUM_WORLDS}" --parallel "${WORLDGEN_PARALLEL}"

	if [[ "${RUN_ANALYZE}" == "1" ]]; then
		run_step "Analyze natural IDs" \
			"${TERRAIN_CMD[@]}" data analyze --source-dir "${ANALYZE_SOURCE_DIR}" --output "${ANALYZE_OUTPUT}"
	else
		log "Skipping analyze stage (RUN_ANALYZE=${RUN_ANALYZE})"
	fi

	run_step "Prepare chunked dataset" \
		"${TERRAIN_CMD[@]}" data prepare \
			--mode chunked \
			--source "${PREP_SOURCE}" \
			--chunk-size "${PREP_CHUNK_SIZE}" \
			--overlap "${PREP_OVERLAP}" \
			--min-diversity "${PREP_MIN_DIVERSITY}" \
			--output-dir "${PREP_OUTPUT_DIR}"

	build_train_args
	run_step "Train model" "${TERRAIN_CMD[@]}" "${TRAIN_ARGS[@]}"

	log "Pipeline complete"
	log "Artifacts: checkpoints/, data/cache/, worldgen/, logs/"
}

run_pipeline
