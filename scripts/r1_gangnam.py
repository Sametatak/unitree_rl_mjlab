# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare and preview a first-pass R1 Gangnam reference; no policy inference."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.assets.robots.unitree_g1.g1_constants import get_spec as get_g1_spec
from src.tasks.tracking.config.r1.env_cfgs import R1_JOINT_NAMES, get_r1_tracking_spec

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/assets/motions/r1/gangnam_g1_source.csv"
OUTPUT = ROOT / "src/assets/motions/r1/gangnam_short.csv"
G1_JOINT_NAMES = (
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
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + tuple(
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
)


def joint_addresses(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    """Return qpos addresses in the explicit CSV joint order."""
    return np.array([model.joint(name).qposadr[0] for name in names])


def set_frame(
    model: mujoco.MjModel, data: mujoco.MjData, frame: np.ndarray, ids: np.ndarray
) -> None:
    """Set a reference frame: root position [m], quaternion xyzw, joints [rad]."""
    data.qpos[:3] = frame[:3]
    data.qpos[3:7] = frame[[6, 3, 4, 5]]  # CSV xyzw -> MuJoCo wxyz.
    data.qpos[ids] = frame[7:]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def foot_geoms(model: mujoco.MjModel) -> np.ndarray:
    """Find the fourteen foot collision capsules used by these robot models."""
    ids = np.array(
        [
            i
            for i in range(model.ngeom)
            if re.fullmatch(r"(left|right)_foot[1-7]_collision", model.geom(i).name)
        ]
    )
    if len(ids) != 14 or np.any(model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError(
            "Expected fourteen foot collision capsules in the source and target models"
        )
    return ids


def sole_height(model: mujoco.MjModel, data: mujoco.MjData, ids: np.ndarray) -> float:
    """Return the lowest collision capsule point [m] in world coordinates."""
    bottom = data.geom_xpos[ids, 2] - model.geom_size[ids, 0]
    bottom -= np.abs(data.geom_xmat[ids, 8]) * model.geom_size[ids, 1]
    return float(bottom.min())


def convert(args: argparse.Namespace) -> None:
    """Map joint angles, fit the R1 waist, and match lowest-foot clearance."""
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; use --overwrite to replace it")
    source = np.loadtxt(args.input, delimiter=",", ndmin=2)
    if source.shape[1] != 36 or not np.isfinite(source).all():
        raise ValueError("Expected finite G1 CSV data with 36 columns")
    if args.start < 0 or args.duration <= 0 or args.fps <= 0:
        raise ValueError("start must be nonnegative; duration and fps must be positive")
    first = round(args.start * args.fps)
    last = first + round(args.duration * args.fps)
    if last >= len(source) or last <= first:
        raise ValueError(
            "Requested clip lies outside the source or contains fewer than two frames"
        )
    clip = source[first : last + 1].copy()
    norms = np.linalg.norm(clip[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Source contains a zero quaternion")
    clip[:, 3:7] /= norms
    clip[:, :2] -= clip[0, :2].copy()
    mapping = [G1_JOINT_NAMES.index(name) for name in R1_JOINT_NAMES]
    output = np.column_stack((clip[:, :7], clip[:, 7:][:, mapping]))

    g1, r1 = get_g1_spec().compile(), get_r1_tracking_spec().compile()
    gdata, rdata = mujoco.MjData(g1), mujoco.MjData(r1)
    gids, rids = (
        joint_addresses(g1, G1_JOINT_NAMES),
        joint_addresses(r1, R1_JOINT_NAMES),
    )
    gfeet, rfeet = foot_geoms(g1), foot_geoms(r1)
    limits = np.array([r1.joint(name).range for name in R1_JOINT_NAMES])
    clipped = np.clip(output[:, 7:], limits[:, 0], limits[:, 1])
    clipped_values = int(np.count_nonzero(np.abs(clipped - output[:, 7:]) > 1e-8))
    output[:, 7:] = clipped
    waist = [
        R1_JOINT_NAMES.index(name) for name in ("waist_roll_joint", "waist_yaw_joint")
    ]
    torso = r1.body("torso_link").id
    source_torso = g1.body("torso_link").id
    errors, offsets = [], []
    for src, dst in zip(clip, output):
        set_frame(g1, gdata, src, gids)
        set_frame(r1, rdata, dst, rids)
        target = Rotation.from_matrix(gdata.xmat[source_torso].reshape(3, 3))

        def residual(angles):
            rdata.qpos[rids[waist]] = angles
            mujoco.mj_forward(r1, rdata)
            actual = Rotation.from_matrix(rdata.xmat[torso].reshape(3, 3))
            return (target.inv() * actual).as_rotvec()

        fit = least_squares(
            residual,
            dst[7:][waist],
            bounds=(limits[waist, 0], limits[waist, 1]),
            max_nfev=40,
        )
        if not fit.success:
            raise RuntimeError(f"Waist orientation fit failed: {fit.message}")
        dst[np.array(waist) + 7] = fit.x
        errors.append(float(np.linalg.norm(residual(fit.x))))
        # Preserve G1's lowest-foot clearance, including flight, rather than pinning every frame to the floor.
        offset = max(0.0, sole_height(g1, gdata, gfeet)) - sole_height(r1, rdata, rfeet)
        dst[2] += offset
        offsets.append(offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, output, delimiter=",", fmt="%.8f")
    report = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "start_s": first / args.fps,
        "duration_s": (last - first) / args.fps,
        "fps": args.fps,
        "frames": len(output),
        "joint_names": R1_JOINT_NAMES,
        "clipped_joint_values": clipped_values,
        "max_torso_error_rad": max(errors),
        "root_height_offset_range_m": [min(offsets), max(offsets)],
        "limitations": "First-pass reference only: no foot XY locking, limb IK, collision avoidance or balance validation.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}\n" + json.dumps(report, indent=2))


def preview(args: argparse.Namespace) -> None:
    """Show a kinematic reference at its recorded rate; no dynamic simulation."""
    import mujoco.viewer

    frames = np.loadtxt(args.input, delimiter=",", ndmin=2)
    report = json.loads(args.input.with_suffix(".json").read_text())
    if frames.shape[1] != 31 or tuple(report["joint_names"]) != R1_JOINT_NAMES:
        raise ValueError("Expected the R1 CSV and its matching joint-order metadata")
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    spec = get_r1_tracking_spec()
    spec.worldbody.add_geom(
        name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    ids = joint_addresses(model, R1_JOINT_NAMES)
    print(
        "KINEMATIC PREVIEW — joint/root poses are prescribed; this is not a balance test."
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.7
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        started = time.monotonic()
        while viewer.is_running():
            index = int(
                (time.monotonic() - started) * report["fps"] * args.speed
            ) % len(frames)
            with viewer.lock():
                set_frame(model, data, frames[index], ids)
                viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--input", type=Path, default=SOURCE)
    convert_parser.add_argument("--output", type=Path, default=OUTPUT)
    convert_parser.add_argument(
        "--start", type=float, default=5.2, help="Clip start [s]."
    )
    convert_parser.add_argument(
        "--duration", type=float, default=3.0, help="Clip duration [s]."
    )
    convert_parser.add_argument(
        "--fps", type=float, default=60.0, help="Source sampling rate [Hz]."
    )
    convert_parser.add_argument("--overwrite", action="store_true")
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("--input", type=Path, default=OUTPUT)
    preview_parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    (convert if args.command == "convert" else preview)(args)
