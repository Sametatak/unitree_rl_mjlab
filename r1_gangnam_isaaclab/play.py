# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an exported mjlab R1 tracking policy in Isaac Lab 3 / Isaac Sim 6."""

import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--bundle", type=Path, default=Path(__file__).parent / "assets/model_3500"
)
parser.add_argument(
    "--robot_usd", default=str(Path(__file__).parent / "assets/robot/r1/r1.usda")
)
parser.add_argument(
    "--once",
    action="store_true",
    help="Pause after one dance instead of resetting and repeating.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
for name in ("config.json", "policy.onnx", "motion.npz"):
    if not (args.bundle / name).is_file():
        parser.error(
            f"Missing {args.bundle / name}; download the complete package (see README.md)."
        )
if not args.headless and args.visualizer is None:
    args.visualizer = ["kit"]
app = AppLauncher(args).app

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def main():
    """Build one physical R1 and run its policy at the trained control rate."""
    cfg = json.loads((args.bundle / "config.json").read_text())
    with np.load(args.bundle / "motion.npz") as source:
        motion = {name: source[name] for name in source.files}
    names = cfg["joint_names"]
    default = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
    default_vel = np.asarray(cfg["default_joint_vel"], dtype=np.float32)
    offset = np.asarray(cfg["action_offset"], dtype=np.float32)
    scale = np.asarray(cfg["action_scale"], dtype=np.float32)
    dt, decimation = cfg["physics_dt"], cfg["decimation"]
    step_dt = dt * decimation
    frames = len(motion["joint_pos"])
    if len(names) != 24 or motion["joint_pos"].shape != (frames, 24):
        raise ValueError("Expected a 24-joint R1 policy and reference.")
    if not np.isclose(float(np.asarray(motion["fps"]).reshape(-1)[0]) * step_dt, 1):
        raise ValueError("Motion FPS and control interval differ.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    policy = ort.InferenceSession(
        str(args.bundle / "policy.onnx"), options, providers=["CPUExecutionProvider"]
    )
    if len(policy.get_inputs()) != 1 or policy.get_inputs()[0].shape != [1, 129]:
        raise ValueError("Expected the R1 actor with one [1, 129] observation input.")
    input_name = policy.get_inputs()[0].name

    def joint_values(key):
        return dict(zip(names, cfg[key]))

    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.6,
        dynamic_friction=0.6,
        restitution=0.0,
        friction_combine_mode="average",
    )
    robot_cfg = ArticulationCfg(
        prim_path="/World/R1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.robot_usd,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            physics_material=material,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76), joint_pos=joint_values("default_joint_pos")
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=joint_values("joint_stiffness"),
                damping=joint_values("joint_damping"),
                effort_limit_sim=joint_values("effort_limit"),
                velocity_limit_sim=1000.0,
                armature=joint_values("armature"),
                friction=joint_values("joint_friction"),
                dynamic_friction=joint_values("joint_friction"),
                viscous_friction=joint_values("passive_damping"),
            )
        },
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args.device, render_interval=decimation)
    )
    ground = sim_utils.GroundPlaneCfg(physics_material=material)
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)
    robot = Articulation(robot_cfg)
    sim.set_camera_view((3.0, 2.5, 1.7), (0.0, 0.0, 0.7))
    sim.reset()
    if set(robot.joint_names) != set(names):
        raise ValueError(
            f"USD joints differ from the training robot: {robot.joint_names}"
        )
    ids = [robot.joint_names.index(name) for name in names]
    torso_id = robot.body_names.index(cfg["anchor_body_name"])
    ref_torso_id = cfg["body_names"].index(cfg["anchor_body_name"])
    ref_root_id = cfg["body_names"].index(cfg["root_body_name"])

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=sim.device)

    def array(value):
        return value.torch.detach().cpu().numpy()[0]

    # Reference quaternions are MuJoCo wxyz; this Isaac Lab installation uses xyzw.
    root_pose = np.concatenate(
        (
            motion["body_pos_w"][0, ref_root_id],
            motion["body_quat_w"][0, ref_root_id, [1, 2, 3, 0]],
        )
    )
    root_velocity = np.concatenate(
        (
            motion["body_lin_vel_w"][0, ref_root_id],
            motion["body_ang_vel_w"][0, ref_root_id],
        )
    )
    ref_rotations = Rotation.from_quat(
        motion["body_quat_w"][:, ref_torso_id][:, [1, 2, 3, 0]]
    )
    print(
        f"Checkpoint: {cfg['run_path']}\nDance: {frames * step_dt:.2f} s; control: {1 / step_dt:g} Hz"
    )
    step = 0
    while app.is_running():
        started = time.monotonic()
        if not sim.is_playing():
            app.update()
            continue
        if step == 0:
            robot.write_root_pose_to_sim_index(root_pose=tensor(root_pose[None]))
            robot.write_root_link_velocity_to_sim_index(
                root_velocity=tensor(root_velocity[None])
            )
            robot.write_joint_position_to_sim_index(
                position=tensor(motion["joint_pos"][0:1]), joint_ids=ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=tensor(motion["joint_vel"][0:1]), joint_ids=ids
            )
            robot.reset()
            robot.update(dt)
            action = np.zeros(24, dtype=np.float32)
        torso = Rotation.from_quat(array(robot.data.body_link_pose_w)[torso_id, 3:7])
        orientation = (torso.inv() * ref_rotations[step]).as_matrix()[:, :2].reshape(-1)
        root = Rotation.from_quat(array(robot.data.root_link_pose_w)[3:7])
        obs = np.concatenate(
            (
                motion["joint_pos"][step],
                motion["joint_vel"][step],
                orientation,
                root.inv().apply(array(robot.data.root_link_vel_w)[3:]),
                array(robot.data.joint_pos)[ids] - default,
                array(robot.data.joint_vel)[ids] - default_vel,
                action,
            )
        ).astype(np.float32)[None]
        action = policy.run(None, {input_name: obs})[0][0]
        if cfg["clip_actions"] is not None:
            action = np.clip(action, -cfg["clip_actions"], cfg["clip_actions"])
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Policy returned invalid joint actions.")
        robot.set_joint_position_target_index(
            target=tensor((offset + scale * action)[None]), joint_ids=ids
        )
        for substep in range(decimation):
            robot.write_data_to_sim()
            sim.step(render=not args.headless and substep == decimation - 1)
            robot.update(dt)
        step = (step + 1) % frames
        if step == 0 and args.once:
            if args.headless:
                break
            print("Dance finished. Simulation paused; Play starts another cycle.")
            sim.pause()
        if not args.headless:
            time.sleep(max(0.0, step_dt - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
