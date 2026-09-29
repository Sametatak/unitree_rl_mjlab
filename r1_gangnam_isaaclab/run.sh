#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R1_ISAACLAB_DIR="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
if [[ ! -f "${R1_ISAACLAB_DIR}/isaaclab.sh" ]]; then
    echo "Isaac Lab bulunamadı. ISAACLAB_PATH=/kurulum/IsaacLab bash run.sh --device cpu" >&2
    exit 1
fi
exec bash "${R1_ISAACLAB_DIR}/isaaclab.sh" -p "${PROJECT_DIR}/play.py" "$@"
