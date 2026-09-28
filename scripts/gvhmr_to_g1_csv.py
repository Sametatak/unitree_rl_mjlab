"""Retarget a GVHMR result to Unitree G1 without opening a viewer."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert GVHMR hmr4d_results.pt into GMR pickle and CSV files."
  )
  parser.add_argument("--gvhmr-result", required=True, type=Path)
  parser.add_argument("--gmr-root", required=True, type=Path)
  parser.add_argument("--output-csv", required=True, type=Path)
  parser.add_argument("--output-pkl", required=True, type=Path)
  parser.add_argument("--target-fps", type=float, default=30.0)
  parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Optional smoke-test limit; omit for the complete motion.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  gvhmr_result = args.gvhmr_result.expanduser().resolve()
  gmr_root = args.gmr_root.expanduser().resolve()
  output_csv = args.output_csv.expanduser().resolve()
  output_pkl = args.output_pkl.expanduser().resolve()

  if not gvhmr_result.is_file():
    raise FileNotFoundError(f"GVHMR result not found: {gvhmr_result}")
  if not (gmr_root / "general_motion_retargeting").is_dir():
    raise FileNotFoundError(f"GMR checkout not found: {gmr_root}")

  sys.path.insert(0, str(gmr_root))
  from general_motion_retargeting import GeneralMotionRetargeting
  from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
  )

  body_models = gmr_root / "assets" / "body_models"
  smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
    str(gvhmr_result), body_models
  )
  frames, aligned_fps = get_gvhmr_data_offline_fast(
    smplx_data,
    body_model,
    smplx_output,
    tgt_fps=args.target_fps,
  )
  if args.max_frames is not None:
    if args.max_frames < 1:
      raise ValueError("--max-frames must be positive")
    frames = frames[: args.max_frames]
  if not frames:
    raise ValueError("GVHMR result contains no motion frames")

  retarget = GeneralMotionRetargeting(
    actual_human_height=human_height,
    src_human="smplx",
    tgt_robot="unitree_g1",
    verbose=False,
    use_velocity_limit=False,
  )

  qpos_rows: list[np.ndarray] = []
  for index, frame in enumerate(frames):
    qpos = np.asarray(retarget.retarget(frame), dtype=np.float64)
    if qpos.shape != (36,):
      raise ValueError(f"Expected 36 G1 qpos values, got {qpos.shape} at frame {index}")
    qpos_rows.append(qpos)
    if (index + 1) % 50 == 0 or index + 1 == len(frames):
      print(f"[GMR] Retargeted {index + 1}/{len(frames)} frames")

  qpos_array = np.stack(qpos_rows)
  root_pos = qpos_array[:, :3]
  root_rot_xyzw = qpos_array[:, 3:7][:, [1, 2, 3, 0]]
  dof_pos = qpos_array[:, 7:]
  motion = np.concatenate((root_pos, root_rot_xyzw, dof_pos), axis=1)

  output_csv.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_csv, motion, delimiter=",")

  output_pkl.parent.mkdir(parents=True, exist_ok=True)
  with output_pkl.open("wb") as file:
    pickle.dump(
      {
        "fps": float(aligned_fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
      },
      file,
    )

  print(f"[GMR] CSV: {output_csv}")
  print(f"[GMR] Pickle: {output_pkl}")
  print(f"[GMR] Frames: {len(motion)}, FPS: {aligned_fps:.3f}")


if __name__ == "__main__":
  main()
