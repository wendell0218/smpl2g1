import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from smpl2g1.generation import adapt_umr_result, canonicalize_motion


class GenerationTest(unittest.TestCase):
    def motion(self):
        rotations = np.zeros((16, 22, 3), dtype=np.float64)
        trans = np.arange(48, dtype=np.float64).reshape(16, 3)
        return {
            "rotations": rotations,
            "trans": trans,
            "betas": np.zeros(10),
            "gender": "neutral",
            "valid": np.ones(16, dtype=bool),
            "fps": 30.0,
        }

    def test_canonical_motion(self):
        fake = types.ModuleType("smpl2g1.motion")
        fake.load_motion = lambda path, fps: self.motion()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"smpl2g1.motion": fake}):
            output = Path(temp) / "motion.npz"
            canonicalize_motion("input.npz", output, "y")
            with np.load(output) as data:
                self.assertEqual(data["poses"].shape, (16, 22, 3))
                self.assertEqual(float(data["fps"]), 30.0)
                self.assertEqual(str(data["output_up"]), "y")

    def test_invalid_frames_are_rejected(self):
        motion = self.motion()
        motion["valid"][3] = False
        fake = types.ModuleType("smpl2g1.motion")
        fake.load_motion = lambda path, fps: motion
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"smpl2g1.motion": fake}):
            with self.assertRaisesRegex(ValueError, "every input frame"):
                canonicalize_motion("input.npz", Path(temp) / "motion.npz", "y")

    def test_adapt_umr_result(self):
        qpos = np.zeros((16, 36), dtype=np.float32)
        qpos[:, 3] = 1
        names = [f"joint_{index}" for index in range(29)]
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw.npz"
            output = Path(temp) / "output.npz"
            np.savez_compressed(
                raw,
                qpos=qpos,
                fps=np.asarray([30], dtype=np.float32),
                robot_joint_names=np.asarray(names, dtype=object),
                frame_ids=np.arange(16),
            )
            adapt_umr_result(raw, output, np.ones(16, dtype=bool), names)
            with np.load(output) as data:
                np.testing.assert_array_equal(data["qpos_36"], qpos)
                self.assertTrue(data["valid"].all())


if __name__ == "__main__":
    unittest.main()
