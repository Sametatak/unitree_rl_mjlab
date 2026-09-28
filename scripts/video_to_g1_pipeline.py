"""Run the video -> GVHMR -> GMR -> G1 NPZ -> training pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoopReport:
  start_frame: int
  end_frame: int
  input_frames: int
  output_frames: int
  joint_error_rad: float
  root_rotation_error_rad: float
  root_position_error_m: float


def command_text(command: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], *, cwd: Path) -> None:
  print(f"\n$ (cd {cwd} && {command_text(command)})", flush=True)
  subprocess.run(command, cwd=cwd, check=True)


def git_revision(repo: Path) -> str | None:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def validate_csv(motion: np.ndarray) -> None:
  if motion.ndim != 2 or motion.shape[1] != 36:
    raise ValueError(
      "Expected G1 CSV shape (frames, 36): root XYZ + quaternion XYZW + 29 joints; "
      f"got {motion.shape}"
    )
  if len(motion) < 2:
    raise ValueError("Motion needs at least two frames")
  if not np.isfinite(motion).all():
    raise ValueError("Motion contains NaN or infinite values")
  quat_norms = np.linalg.norm(motion[:, 3:7], axis=1)
  if np.any(quat_norms < 1e-8):
    raise ValueError("Motion contains an invalid zero-length root quaternion")


def find_loop_bounds(
  motion: np.ndarray,
  *,
  fps: float,
  min_loop_seconds: float,
  search_start_frames: int,
) -> tuple[int, int]:
  min_frames = max(2, round(min_loop_seconds * fps))
  if len(motion) <= min_frames:
    raise ValueError(
      f"Motion has {len(motion)} frames but loop search needs more than {min_frames}"
    )

  last_start = min(search_start_frames, len(motion) - min_frames - 1)
  best: tuple[float, int, int] | None = None
  for start in range(last_start + 1):
    start_rotation = Rotation.from_quat(motion[start, 3:7])
    for end in range(start + min_frames, len(motion)):
      joint_error = float(np.mean(np.abs(motion[start, 7:] - motion[end, 7:])))
      rotation_error = float(
        (start_rotation.inv() * Rotation.from_quat(motion[end, 3:7])).magnitude()
      )
      score = joint_error + 0.20 * rotation_error
      if best is None or score < best[0]:
        best = (score, start, end)

  if best is None:
    raise RuntimeError("Could not find loop boundaries")
  return best[1], best[2]


def make_loop(
  motion: np.ndarray,
  *,
  start: int,
  end: int,
  blend_frames: int,
) -> tuple[np.ndarray, LoopReport]:
  validate_csv(motion)
  if not 0 <= start < end < len(motion):
    raise ValueError(f"Invalid loop bounds [{start}, {end}] for {len(motion)} frames")

  sequence = motion[start : end + 1].copy()
  if blend_frames < 2 or 2 * blend_frames >= len(sequence):
    raise ValueError("--blend-frames must be >= 2 and smaller than half the cropped loop")

  xy_drift = sequence[-1, :2] - sequence[0, :2]
  progress = np.linspace(0.0, 1.0, len(sequence))[:, None]
  sequence[:, :2] -= progress * xy_drift

  head = sequence[:blend_frames].copy()
  tail = sequence[-blend_frames:].copy()
  body = sequence[blend_frames:-blend_frames].copy()
  blend = tail.copy()

  for index in range(blend_frames):
    t = index / (blend_frames - 1)
    alpha = 3 * t * t - 2 * t * t * t
    blend[index, :3] = (1 - alpha) * tail[index, :3] + alpha * head[index, :3]
    rotations = Rotation.from_quat([tail[index, 3:7], head[index, 3:7]])
    blend[index, 3:7] = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0]
    blend[index, 7:] = (1 - alpha) * tail[index, 7:] + alpha * head[index, 7:]

  loop = np.vstack((body, blend))
  validate_csv(loop)
  root_rotation_error = float(
    (Rotation.from_quat(loop[-1, 3:7]).inv() * Rotation.from_quat(loop[0, 3:7])).magnitude()
  )
  report = LoopReport(
    start_frame=start,
    end_frame=end,
    input_frames=len(motion),
    output_frames=len(loop),
    joint_error_rad=float(np.mean(np.abs(loop[-1, 7:] - loop[0, 7:]))),
    root_rotation_error_rad=root_rotation_error,
    root_position_error_m=float(np.linalg.norm(loop[-1, :3] - loop[0, :3])),
  )
  return loop, report


def find_gvhmr_result(output_root: Path, video_stem: str) -> Path:
  expected = output_root / video_stem / "hmr4d_results.pt"
  if expected.is_file():
    return expected
  matches = list(output_root.rglob("hmr4d_results.pt"))
  if len(matches) == 1:
    return matches[0]
  raise FileNotFoundError(
    f"Expected one GVHMR result under {output_root}, found {len(matches)}"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert an MP4 into a looped G1 motion NPZ and start tracking training."
  )
  parser.add_argument("video", type=Path, help="Single-person halay MP4 input")
  parser.add_argument("--motion-name", default=None, help="Output stem; default: <video>_loop")
  parser.add_argument("--gvhmr-root", type=Path, default=Path.home() / "GVHMR-blackwell")
  parser.add_argument("--gmr-root", type=Path, default=Path.home() / "GMR")
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument(
    "--motion-dir",
    type=Path,
    default=REPO_ROOT / "src" / "assets" / "motions" / "g1",
    help="Destination for the final CSV and NPZ.",
  )
  parser.add_argument(
    "--gvhmr-result",
    type=Path,
    default=None,
    help="Use an existing hmr4d_results.pt and skip video recovery.",
  )
  parser.add_argument("--moving-camera", action="store_true")
  parser.add_argument("--gvhmr-camera", choices=("simplevo", "dpvo", "dust3r", "vggt"), default="simplevo")
  parser.add_argument("--detector-confidence", type=float, default=0.05)
  parser.add_argument("--render-gvhmr", action="store_true")
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--loop-start", type=int, default=None)
  parser.add_argument("--loop-end", type=int, default=None)
  parser.add_argument("--min-loop-seconds", type=float, default=3.0)
  parser.add_argument("--search-start-frames", type=int, default=100)
  parser.add_argument("--blend-frames", type=int, default=15)
  parser.add_argument("--no-loop", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--train-iterations", type=int, default=None)
  parser.add_argument("--train-extra-arg", action="append", default=[])
  parser.add_argument("--prepare-only", action="store_true", help="Stop after creating the NPZ")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  video = args.video.expanduser().resolve()
  gvhmr_root = args.gvhmr_root.expanduser().resolve()
  gmr_root = args.gmr_root.expanduser().resolve()
  if not video.is_file() and args.gvhmr_result is None:
    raise FileNotFoundError(f"Input video not found: {video}")
  if not (gvhmr_root / "bin" / "gvhmr").is_file() and args.gvhmr_result is None:
    raise FileNotFoundError(f"GVHMR executable not found: {gvhmr_root / 'bin/gvhmr'}")
  gmr_python = gmr_root / ".venv" / "bin" / "python"
  if not gmr_python.is_file():
    raise FileNotFoundError(f"GMR Python environment not found: {gmr_python}")

  default_name = video.stem if video.stem.endswith("_loop") else f"{video.stem}_loop"
  motion_name = args.motion_name or default_name
  if not motion_name or Path(motion_name).name != motion_name:
    raise ValueError("--motion-name must be a filename stem without directory separators")

  work_dir = (args.work_dir or REPO_ROOT / "pipeline_runs" / motion_name).expanduser().resolve()
  work_dir.mkdir(parents=True, exist_ok=True)
  gvhmr_output_root = work_dir / "gvhmr"

  if args.gvhmr_result is not None:
    gvhmr_result = args.gvhmr_result.expanduser().resolve()
    if not gvhmr_result.is_file():
      raise FileNotFoundError(f"GVHMR result not found: {gvhmr_result}")
  else:
    gvhmr_command = [
      str(gvhmr_root / "bin" / "gvhmr"),
      "demo",
      str(video),
      "--output-root",
      str(gvhmr_output_root),
      "--set",
      f"detector.conf={args.detector_confidence}",
    ]
    if args.moving_camera:
      gvhmr_command.extend(("--camera", args.gvhmr_camera))
    else:
      gvhmr_command.append("--static-cam")
    if args.render_gvhmr:
      gvhmr_command.append("--smplx")
    else:
      gvhmr_command.append("--no-render")
    run(gvhmr_command, cwd=gvhmr_root)
    gvhmr_result = find_gvhmr_result(gvhmr_output_root, video.stem)

  raw_csv = work_dir / f"{motion_name}_raw.csv"
  raw_pkl = work_dir / f"{motion_name}_retargeted.pkl"
  run(
    [
      str(gmr_python),
      str(REPO_ROOT / "scripts" / "gvhmr_to_g1_csv.py"),
      "--gvhmr-result",
      str(gvhmr_result),
      "--gmr-root",
      str(gmr_root),
      "--output-csv",
      str(raw_csv),
      "--output-pkl",
      str(raw_pkl),
      "--target-fps",
      str(args.input_fps),
    ],
    cwd=gmr_root,
  )

  raw_motion = np.loadtxt(raw_csv, delimiter=",")
  validate_csv(raw_motion)
  report: LoopReport | None = None
  if args.no_loop:
    final_motion = raw_motion
  else:
    if (args.loop_start is None) != (args.loop_end is None):
      raise ValueError("Use --loop-start and --loop-end together")
    if args.loop_start is None:
      loop_start, loop_end = find_loop_bounds(
        raw_motion,
        fps=args.input_fps,
        min_loop_seconds=args.min_loop_seconds,
        search_start_frames=args.search_start_frames,
      )
    else:
      loop_start, loop_end = args.loop_start, args.loop_end
    final_motion, report = make_loop(
      raw_motion,
      start=loop_start,
      end=loop_end,
      blend_frames=args.blend_frames,
    )
    print(
      "[Loop] "
      f"frames {report.start_frame}:{report.end_frame}, output={report.output_frames}, "
      f"joint seam={report.joint_error_rad:.4f} rad, "
      f"root seam={np.degrees(report.root_rotation_error_rad):.2f} deg"
    )

  motion_dir = args.motion_dir.expanduser().resolve()
  motion_dir.mkdir(parents=True, exist_ok=True)
  final_csv = motion_dir / f"{motion_name}.csv"
  final_npz = motion_dir / f"{motion_name}.npz"
  np.savetxt(final_csv, final_motion, delimiter=",")

  run(
    [
      sys.executable,
      str(REPO_ROOT / "scripts" / "csv_to_npz.py"),
      "--input-file",
      str(final_csv),
      "--output-name",
      str(final_npz),
      "--input-fps",
      str(args.input_fps),
      "--output-fps",
      str(args.output_fps),
      "--robot",
      "g1",
      "--device",
      args.device,
    ],
    cwd=REPO_ROOT,
  )
  if not final_npz.is_file():
    raise FileNotFoundError(f"CSV conversion did not create {final_npz}")

  metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "video": str(video),
    "motion_name": motion_name,
    "gvhmr_result": str(gvhmr_result),
    "gvhmr_revision": git_revision(gvhmr_root),
    "gmr_revision": git_revision(gmr_root),
    "unitree_rl_mjlab_revision": git_revision(REPO_ROOT),
    "raw_csv": str(raw_csv),
    "final_csv": str(final_csv),
    "final_npz": str(final_npz),
    "loop": asdict(report) if report is not None else None,
    "input_fps": args.input_fps,
    "output_fps": args.output_fps,
  }
  metadata_path = work_dir / "pipeline_run.json"
  metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
  print(f"\n[Done] Motion NPZ: {final_npz}")
  print(f"[Done] Run metadata: {metadata_path}")

  if args.prepare_only:
    print("[Done] --prepare-only selected; training was not started.")
    return

  train_command = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "train.py"),
    "Unitree-G1-Tracking-No-State-Estimation",
    "--motion-file",
    str(final_npz),
    f"--env.scene.num-envs={args.num_envs}",
  ]
  if args.train_iterations is not None:
    train_command.append(f"--agent.max-iterations={args.train_iterations}")
  train_command.extend(args.train_extra_arg)
  run(train_command, cwd=REPO_ROOT)


if __name__ == "__main__":
  main()
