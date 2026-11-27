#!/usr/bin/env bash
# Freeze the data-contract chain:
#   field_spec → vocabulary → inner_table → row_subset
#
# Does not create a venv. Uses project-root .venv python
# (Scripts/python.exe on Windows, bin/python on Linux). Never uv run.
# Never sets PYTHONPATH. ccf_tabformer_dl must already be installed.
#
# Usage (from anywhere):
#   bash scripts/data/bash_scripts/run_data_contract.sh
#   bash scripts/data/bash_scripts/run_data_contract.sh --force
#
# --force is forwarded to each define_* script (cleanup then rewrite).
# process() still has no overwrite. Without --force, a named collection
# alias that already exists is skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
SETUP="${REPO}/scripts/data"
VENV="${CCF_TABFORMER_VENV:-${REPO}/.venv}"
PKG_ART="${REPO}/ccf_tabformer_dl/src/ccf_tabformer_dl/package_artifacts"

FIELD_SPEC_NAME="declared_v1"
VOCABULARY_NAME="universal_digits_v1"
INNER_TABLE_NAME="fixed_width_v1"
ROW_SUBSET_NAME="alarm_label_1k_seed0"

FIELD_SPEC_ALIAS="${PKG_ART}/field_spec/collections/${FIELD_SPEC_NAME}.json"
VOCABULARY_ALIAS="${PKG_ART}/vocabulary/collections/${VOCABULARY_NAME}.json"
INNER_TABLE_ALIAS="${PKG_ART}/inner_table/collections/${INNER_TABLE_NAME}.json"
ROW_SUBSET_ALIAS="${PKG_ART}/row_subset/collections/${ROW_SUBSET_NAME}.json"

usage() {
  echo "Usage: $0 [--force]" >&2
  exit 1
}

FORCE=0
for arg in "$@"; do
  case "${arg}" in
    --force)
      FORCE=1
      ;;
    *)
      usage
      ;;
  esac
done

PYTHON=""
for candidate in \
    "${VENV}/Scripts/python.exe" \
    "${VENV}/Scripts/python" \
    "${VENV}/bin/python" \
    "${VENV}/bin/python3"
do
  if [[ -f "${candidate}" ]]; then
    PYTHON="${candidate}"
    break
  fi
done
if [[ -z "${PYTHON}" ]]; then
  echo "error: no Python in ${VENV}" >&2
  echo "checked Windows: ${VENV}/Scripts/python.exe" >&2
  echo "checked Linux:   ${VENV}/bin/python" >&2
  echo "do not create a new venv; point CCF_TABFORMER_VENV at the existing one" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

if ! "${PYTHON}" -c "import ccf_tabformer_dl" >/dev/null 2>&1; then
  echo "error: ccf_tabformer_dl is not importable from ${PYTHON}" >&2
  echo "activate the project .venv, then: uv sync --active" >&2
  exit 1
fi

run_py() {
  echo ""
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "\$ ${PYTHON} $* --force"
    "${PYTHON}" "$@" --force
  else
    echo "\$ ${PYTHON} $*"
    "${PYTHON}" "$@"
  fi
}

run_step() {
  local label="$1"
  local alias="$2"
  shift 2
  if [[ "${FORCE}" -eq 0 && -f "${alias}" ]]; then
    echo "=== skip ${label} (already frozen as ${alias}) ==="
    return 0
  fi
  echo "=== ${label} ==="
  run_py "$@"
}

cd "${REPO}"

echo "============================================================"
echo "data contract: ${FIELD_SPEC_NAME} → ${VOCABULARY_NAME} → ${INNER_TABLE_NAME} → ${ROW_SUBSET_NAME}"
echo "python=${PYTHON}"
echo "============================================================"

run_step "field_spec ${FIELD_SPEC_NAME}" "${FIELD_SPEC_ALIAS}" \
  "${SETUP}/define_field_spec_collection.py" \
  --name "${FIELD_SPEC_NAME}"

run_step "vocabulary ${VOCABULARY_NAME}" "${VOCABULARY_ALIAS}" \
  "${SETUP}/define_vocabulary_collection.py" \
  --parent "${FIELD_SPEC_NAME}" \
  --name "${VOCABULARY_NAME}"

run_step "inner_table ${INNER_TABLE_NAME}" "${INNER_TABLE_ALIAS}" \
  "${SETUP}/define_inner_table_collection.py" \
  --parent "${VOCABULARY_NAME}" \
  --name "${INNER_TABLE_NAME}"

run_step "row_subset ${ROW_SUBSET_NAME}" "${ROW_SUBSET_ALIAS}" \
  "${SETUP}/define_row_subset_collection.py" \
  --parent "${INNER_TABLE_NAME}" \
  --name "${ROW_SUBSET_NAME}"

echo "============================================================"
echo "data contract done"
echo "============================================================"
