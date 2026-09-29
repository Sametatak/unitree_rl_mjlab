# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an R1 tracking checkpoint and its reference for the Isaac Lab player."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401


class ConfigLoader(yaml.SafeLoader):
    """Read saved YAML as data without importing or executing its Python tags."""


def _python_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


def _merge(target, saved):
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def main():
    """Create policy.onnx, motion.npz and config.json without a training rollout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motion_file", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    saved = yaml.load(
        (checkpoint.parent / "params/env.yaml").read_text(), Loader=ConfigLoader
    )
    saved_agent = yaml.load(
        (checkpoint.parent / "params/agent.yaml").read_text(), Loader=ConfigLoader
    )
    task = "Unitree-R1-Tracking-No-State-Estimation"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.events = {}
    cfg.decimation = saved["decimation"]
    cfg.sim.mujoco.timestep = saved["sim"]["mujoco"]["timestep"]
    robot_cfg = cfg.scene.entities["robot"]
    saved_robot = saved["scene"]["entities"]["robot"]
    robot_cfg.init_state = replace(robot_cfg.init_state, **saved_robot["init_state"])
    saved_actuators = saved_robot["articulation"]["actuators"]
    if len(robot_cfg.articulation.actuators) != len(saved_actuators):
        raise ValueError("Actuator configuration changed since training.")
    robot_cfg.articulation.actuators = tuple(
        replace(
            current, **{k: v for k, v in stored.items() if k != "transmission_type"}
        )
        for current, stored in zip(robot_cfg.articulation.actuators, saved_actuators)
    )
    for key in (
        "scale",
        "offset",
        "use_default_offset",
        "clip",
        "preserve_order",
        "actuator_names",
    ):
        setattr(cfg.actions["joint_pos"], key, saved["actions"]["joint_pos"][key])
    expected = [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    actor = saved["observations"]["actor"]
    if list(actor["terms"]) != expected or actor.get("history_length"):
        raise ValueError(
            "This player requires the 129-value R1 no-state-estimation observation."
        )
    for term in actor["terms"].values():
        if any(
            term.get(key)
            for key in ("history_length", "delay_max_lag", "clip", "scale")
        ):
            raise ValueError(
                "Observation history, delays, clipping or scaling need a matching player."
            )
    motion_cfg = cfg.commands["motion"]
    motion_file = args.motion_file or Path(saved["commands"]["motion"]["motion_file"])
    motion_file = motion_file.expanduser().resolve(strict=True)
    motion_cfg.motion_file = str(motion_file)
    motion_cfg.body_names = tuple(saved["commands"]["motion"]["body_names"])
    motion_cfg.anchor_body_name = saved["commands"]["motion"]["anchor_body_name"]
    agent = asdict(load_rl_cfg(task))
    _merge(agent, saved_agent)
    agent["logger"] = "tensorboard"
    agent["resume"] = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
        runner = load_runner_cls(task)(wrapped, agent, device="cpu")
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
        )
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        if (
            list(action.target_names) != list(robot.joint_names)
            or len(robot.joint_names) != 24
        ):
            raise ValueError(
                "Player expects all 24 actions in the observation joint order."
            )
        metadata = get_base_metadata(env, str(checkpoint))
        metadata["default_joint_vel"] = robot.data.default_joint_vel[0].cpu().tolist()
        metadata["action_offset"] = action.offset[0].cpu().tolist()
        metadata["clip_actions"] = agent["clip_actions"]
        metadata["action_clip"] = saved["actions"]["joint_pos"]["clip"]
        if metadata["action_clip"] is not None:
            raise ValueError(
                "Per-joint action clipping requires a matching Isaac player."
            )
        model = env.sim.mj_model
        actuators = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
        ctrl_ids = [actuators[name] for name in robot.joint_names]
        dof_ids = [
            int(model.joint(f"robot/{name}").dofadr[0]) for name in robot.joint_names
        ]
        metadata["effort_limit"] = (
            np.abs(model.actuator_forcerange[ctrl_ids]).max(axis=1).tolist()
        )
        metadata["armature"] = model.dof_armature[dof_ids].tolist()
        metadata["joint_friction"] = model.dof_frictionloss[dof_ids].tolist()
        metadata["passive_damping"] = model.dof_damping[dof_ids].tolist()
        metadata.update(
            physics_dt=cfg.sim.mujoco.timestep,
            decimation=cfg.decimation,
            body_names=list(motion_cfg.body_names),
            anchor_body_name=motion_cfg.anchor_body_name,
            root_body_name="pelvis",
            motion_source=str(motion_file),
            motion_sha256=hashlib.sha256(motion_file.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        with np.load(motion_file) as source:
            fps = float(np.asarray(source["fps"]).reshape(-1)[0])
        if not np.isclose(fps * cfg.decimation * cfg.sim.mujoco.timestep, 1.0):
            raise ValueError("Reference FPS must match the training control frequency.")
        motion = env.command_manager.get_term("motion").motion
        arrays = {
            key: getattr(motion, key).cpu().numpy()
            for key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(output), "policy.onnx")
        np.savez_compressed(output / "motion.npz", fps=fps, **arrays)
        (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Exported {checkpoint.name} -> {output}")
        print(f"Reference: {arrays['joint_pos'].shape[0] / fps:.2f} s; {fps:g} Hz")
    finally:
        env.close()


if __name__ == "__main__":
    main()
