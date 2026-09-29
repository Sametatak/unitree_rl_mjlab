# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Use the existing tracking PPO configuration for R1."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_r1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create an R1 runner with a separate checkpoint directory."""
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "r1_tracking"
    cfg.logger = "tensorboard"
    return cfg
