#!/usr/bin/env bash
set -Eeuo pipefail
root=${1:?Run root required}
mkdir -p "$root/evidence"
exec >> "$root/evidence/tmux-entry.log" 2>&1
printf '%s\n' "$$" > "$root/evidence/entry.PID"
trap 'status=$?; printf "%s\n" "$status" > "$root/evidence/tmux-entry.EXIT"; if (( status != 0 )); then printf "entry failure exit=%s at %s\n" "$status" "$(date --iso-8601=seconds)" >> "$root/evidence/entry-failures.log"; fi' EXIT
printf 'entry start %s pid=%s\n' "$(date --iso-8601=seconds)" "$$"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Conda activation hooks legitimately reference unset shell variables.
set +u
source /opt/miniforge/etc/profile.d/conda.sh
conda activate layerpano3d-blackwell
set -u
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$root/evidence/tmp"
mkdir -p "$TMPDIR"
python -u scripts/curated_two.py --root "$root" --run --owner assistant >> "$root/evidence/driver.log" 2>&1 &
driver=$!
printf '%s\n' "$driver" > "$root/evidence/driver.PID"
wait "$driver"
