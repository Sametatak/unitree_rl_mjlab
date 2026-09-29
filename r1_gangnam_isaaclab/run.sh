#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R1_ISAACLAB_DIR="${ISAACLAB_PATH:-${HOME}/IsaacLab}"

# Prefer an explicitly selected Python, then the active environment, followed
# by the local G1 IsaacLab project's environment used on this workstation.
python_candidates=(
    "${ISAACLAB_PYTHON:-}"
    "${VIRTUAL_ENV:-}/bin/python"
    "${HOME}/isaac-lab/unitree_g1_tracking_isaaclab/.venv/bin/python"
    "${R1_ISAACLAB_DIR}/.venv/bin/python"
)
for python_exe in "${python_candidates[@]}"; do
    if [[ -x "${python_exe}" ]] && "${python_exe}" -c "import isaaclab, isaacsim" >/dev/null 2>&1; then
        exec "${python_exe}" "${PROJECT_DIR}/play.py" "$@"
    fi
done

echo "Isaac Lab + Isaac Sim Python ortamı bulunamadı." >&2
echo "ISAACLAB_PYTHON=/yol/.venv/bin/python bash run.sh --device cpu" >&2
exit 1
