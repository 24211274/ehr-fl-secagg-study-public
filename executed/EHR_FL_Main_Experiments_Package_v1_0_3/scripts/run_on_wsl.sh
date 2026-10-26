#!/usr/bin/env bash
# New campaign for preparation; reuse sealed source and accepted venv for later actions.
set -Eeuo pipefail
PACKAGE=$(realpath -- "$1")
MODE=${2:-}
if [[ $(id -u) == 0 ]]; then echo 'STOP: do not use root/sudo.' >&2; exit 1; fi
if [[ "$MODE" == Prepare || "$MODE" == PrepareRepair ]]; then
  EXTRA=()
  if [[ "$MODE" == PrepareRepair ]]; then
    [[ $# -eq 3 ]] || { echo 'Usage: package PrepareRepair reviewed-parent-campaign'; exit 2; }
    PARENT=$(realpath -- "$3")
    [[ -f "$PARENT/origins.json" ]] || { echo 'STOP: parent origins unavailable'; exit 1; }
    mapfile -t ORIGINS < <(python3 -I -B -c 'import json,sys; v=json.load(open(sys.argv[1])); print(v["inference_run"]); print(v["accepted_run"]); print(v["preprocessing_run"])' "$PARENT/origins.json")
    [[ ${#ORIGINS[@]} -eq 3 ]] || { echo 'STOP: parent origins are incomplete'; exit 1; }
    INFERENCE=$(realpath -- "${ORIGINS[0]}"); PATCHED=$(realpath -- "${ORIGINS[1]}"); PRE=$(realpath -- "${ORIGINS[2]}")
    EXTRA=(--parent-campaign "$PARENT")
  else
  [[ $# -eq 5 ]] || { echo 'Usage: package Prepare inference-run patched-run preprocessing-run'; exit 2; }
  INFERENCE=$(realpath -- "$3"); PATCHED=$(realpath -- "$4"); PRE=$(realpath -- "$5")
  fi
  PY="$PATCHED/.venv_full_candidate/bin/python"
  [[ -x "$PY" ]] || { echo 'STOP: accepted Python does not exist; no installation or fallback.'; exit 1; }
  RID=$(date -u +%Y%m%dT%H%M%SZ)_$("$PY" -c 'import uuid; print(uuid.uuid4().hex[:8])')
  CAMPAIGN="$HOME/EHR_FL_Main_v1/campaigns/$RID"
  mkdir -p "$HOME/EHR_FL_Main_v1/campaigns"; mkdir "$CAMPAIGN"
  "$PY" -I -B - "$PACKAGE" "$CAMPAIGN/source" <<'PY'
import shutil,sys
from pathlib import Path
source,dest=map(Path,sys.argv[1:])
shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*Report_*.zip'))
PY
  SOURCE="$CAMPAIGN/source"
  set +e
  "$PY" -I -B "$SOURCE/scripts/main_cli.py" "$MODE" --campaign "$CAMPAIGN" --inference-run "$INFERENCE" --accepted-run "$PATCHED" --preprocessing-run "$PRE" "${EXTRA[@]}"
  RC=$?
  set -e
elif [[ "$MODE" == TrainNextBlock || "$MODE" == RetryBlock || "$MODE" == Evaluate ]]; then
  [[ $# -ge 4 ]] || { echo 'Usage: package mode campaign reviewed-authorisation [benchmark-source]'; exit 2; }
  CAMPAIGN=$(realpath -- "$3"); AUTH=$(realpath -- "$4"); SOURCE="$CAMPAIGN/source"
  [[ -f "$CAMPAIGN/origins.json" ]] || { echo 'STOP: campaign origins unavailable'; exit 1; }
  # Use only the accepted path saved during preparation; no shell evaluation of JSON.
  PATCHED=$(python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["accepted_run"])' "$CAMPAIGN/origins.json")
  PY="$PATCHED/.venv_full_candidate/bin/python"
  ARGS=("$MODE" --campaign "$CAMPAIGN" --authorisation "$AUTH")
  if [[ "$MODE" == Evaluate ]]; then [[ $# -eq 5 ]] || exit 2; ARGS+=(--benchmark-source "$5"); fi
  set +e
  "$PY" -I -B "$SOURCE/scripts/main_cli.py" "${ARGS[@]}"
  RC=$?
  set -e
else
  echo 'STOP: choose Prepare, PrepareRepair, TrainNextBlock, RetryBlock, or Evaluate explicitly.' >&2; exit 2
fi
# Share reports only. Never overwrite a prior Windows report or export checkpoints/secrets.
"$PY" -I -B - "$CAMPAIGN" "$PACKAGE" <<'PY'
import sys,shutil,hashlib
from pathlib import Path
campaign,target=map(Path,sys.argv[1:])
for p in sorted(campaign.glob('*Report_*.zip')):
    dst=target/p.name
    if dst.exists():
        if hashlib.sha256(dst.read_bytes()).digest()!=hashlib.sha256(p.read_bytes()).digest():raise SystemExit('STOP: conflicting existing report')
        continue
    with p.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    print('WINDOWS_REPORT_FILENAME='+p.name)
PY
if [[ $RC -ne 0 ]]; then echo 'STOP: preserve the report; no retuning/reinstallation/automatic retry.'; fi
exit "$RC"
