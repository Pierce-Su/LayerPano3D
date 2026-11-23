#!/usr/bin/env bash
# Only Assistant owns GPU launch; this script creates no cron/monitor loop.
set -Eeuo pipefail
root=${1:?Usage: bash scripts/launch_curated_two.sh ABSOLUTE_RUN_ROOT}
[[ "$root" =~ ^/mnt/NAS/E3DQA_project/assets/LayerPano3D/[A-Za-z0-9_.-]+$ ]] || { printf 'Unexpected output root\n' >&2; exit 64; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/miniforge/etc/profile.d/conda.sh
conda activate layerpano3d-blackwell
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/curated_two.py --root "$root" --check-ready
exec 9>"$root/evidence/launch.lock"
flock -n 9 || { printf 'Another launch owns the lock\n' >&2; exit 75; }
session=layerpano3d-curated-two
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session already exists; inspect it, do not launch a duplicate: %s\n' "$session" >&2
    exit 75
fi
python scripts/curated_two.py --root "$root" --gate
# Driver rechecks GPU idleness itself; no claim that the preceding snapshot reserves a GPU.
tmux new-session -d -s "$session" -c "$repo" "bash '$repo/scripts/curated_tmux_entry.sh' '$root'"
tmux list-panes -t "$session" -F 'session=#{session_name} pane_pid=#{pane_pid} command=#{pane_current_command}'
printf 'Created %s. Exact driver PID: %s/evidence/driver.PID; log: %s/evidence/driver.log\n' "$session" "$root" "$root"
