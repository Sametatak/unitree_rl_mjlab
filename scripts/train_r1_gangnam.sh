#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
R1_PYTHON="${R1_PYTHON:-python}"
R1_CSV="src/assets/motions/r1/gangnam_short.csv"
R1_NPZ="src/assets/motions/r1/gangnam_short.npz"
if [[ ! -f "$R1_CSV" ]]; then
    echo "Önce scripts/r1_gangnam.py convert komutuyla referansı oluştur." >&2
    exit 1
fi
"$R1_PYTHON" scripts/csv_to_npz.py --robot r1 \
    --input-file "$R1_CSV" --output-name gangnam_short.npz \
    --input-fps 60 --output-fps 50 --device cpu
exec "$R1_PYTHON" scripts/train.py Unitree-R1-Tracking-No-State-Estimation \
    --motion-file "$R1_NPZ" --env.scene.num-envs "${R1_NUM_ENVS:-256}" "$@"
