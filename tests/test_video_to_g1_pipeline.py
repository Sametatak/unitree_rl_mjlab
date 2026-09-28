import unittest

import numpy as np

from scripts.video_to_g1_pipeline import find_loop_bounds, make_loop, validate_csv


def sample_motion(frames: int = 140) -> np.ndarray:
  motion = np.zeros((frames, 36), dtype=np.float64)
  motion[:, 0] = np.linspace(0.0, 1.0, frames)
  motion[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
  phase = np.linspace(0.0, 4.0 * np.pi, frames)
  motion[:, 7:] = np.sin(phase)[:, None]
  return motion


class VideoToG1PipelineTest(unittest.TestCase):
  def test_loop_has_expected_shape_and_finite_values(self):
    motion = sample_motion()
    loop, report = make_loop(motion, start=0, end=100, blend_frames=10)

    self.assertEqual(loop.shape, (91, 36))
    self.assertEqual(report.output_frames, 91)
    self.assertTrue(np.isfinite(loop).all())

  def test_automatic_bounds_respect_minimum_duration(self):
    motion = sample_motion()
    start, end = find_loop_bounds(
      motion,
      fps=30.0,
      min_loop_seconds=3.0,
      search_start_frames=20,
    )

    self.assertGreaterEqual(end - start, 90)

  def test_wrong_column_count_is_rejected(self):
    with self.assertRaises(ValueError):
      validate_csv(np.zeros((10, 35)))


if __name__ == "__main__":
  unittest.main()
