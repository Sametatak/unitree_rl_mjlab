# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree R1 flat tracking environment configurations."""

import xml.etree.ElementTree as ET

import mujoco

from src.assets.robots.unitree_r1.r1_constants import (
    R1_ACTION_SCALE,
    get_r1_robot_cfg,
    R1_XML,
    get_assets,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


R1_JOINT_NAMES = (
    tuple(
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    )
    + ("waist_roll_joint", "waist_yaw_joint")
    + tuple(
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
        )
    )
)


def get_r1_tracking_spec() -> mujoco.MjSpec:
    """Load R1 while omitting contact exclusions for nonexistent bodies."""
    root = ET.parse(R1_XML).getroot()
    bodies = {body.get("name") for body in root.iter("body")}
    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact.findall("exclude")):
            if any(exclude.get(key) not in bodies for key in ("body1", "body2")):
                contact.remove(exclude)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.assets = get_assets(spec.meshdir)
    return spec


def unitree_r1_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree R1 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_r1_robot_cfg()}
    cfg.scene.entities["robot"].spec_fn = get_r1_tracking_spec
    cfg.sim.nconmax = 48
    cfg.sim.contact_sensor_maxmatch = 500

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = R1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_link",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    )

    cfg.viewer.body_name = "torso_link"

    # Modify observations if we don't have state estimation.
    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
