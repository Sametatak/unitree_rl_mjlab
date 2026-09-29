# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Register R1 motion tracking tasks with the standard train/play scripts."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_r1_flat_tracking_env_cfg
from .rl_cfg import unitree_r1_tracking_ppo_runner_cfg

for has_state_estimation, suffix in ((True, ""), (False, "-No-State-Estimation")):
    register_mjlab_task(
        task_id=f"Unitree-R1-Tracking{suffix}",
        env_cfg=unitree_r1_flat_tracking_env_cfg(
            has_state_estimation=has_state_estimation
        ),
        play_env_cfg=unitree_r1_flat_tracking_env_cfg(
            has_state_estimation=has_state_estimation, play=True
        ),
        rl_cfg=unitree_r1_tracking_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
