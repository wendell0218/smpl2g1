import unittest

import numpy as np

from smpl2g1.pipeline import adaptive_wrist_filter, valid_runs
from smpl2g1.refinement import blend_upper_body, project_joint_feasibility, smooth_root_pose


class PipelineTest(unittest.TestCase):
    def test_valid_runs(self):
        valid = np.asarray([True, True, False, True, False, True, True])
        self.assertEqual(valid_runs(valid), [(0, 2), (3, 4), (5, 7)])

    def test_wrist_filter_noop(self):
        qpos = np.zeros((30, 36))
        limits = np.column_stack([-np.ones(29), np.ones(29)])
        output, cutoff = adaptive_wrist_filter(qpos, 30.0, limits)
        self.assertIsNone(cutoff)
        np.testing.assert_array_equal(output, qpos)

    # <modify>+Verify position margins and adjacent-frame velocity boxes together.
    def test_joint_feasibility_projection(self):
        qpos = np.zeros((5, 36))
        qpos[1, 7:] = 2.0
        qpos[2, 7:] = -2.0
        limits = np.column_stack([-np.ones(29), np.ones(29)])
        output, lower, upper = project_joint_feasibility(qpos, limits, np.full(29, 0.1))
        self.assertTrue((output[:, 7:] >= lower - 1e-12).all())
        self.assertTrue((output[:, 7:] <= upper + 1e-12).all())
        self.assertTrue((np.abs(np.diff(output[:, 7:], axis=0)) <= 0.1 + 1e-12).all())

    # <modify>+Verify that root smoothing leaves joints unchanged and preserves unit quaternions.
    def test_refinement_smoothing(self):
        qpos = np.zeros((61, 36))
        qpos[:, 3] = 1.0
        qpos[:, 0] = (-1.0) ** np.arange(61)
        qpos[:, 7] = np.arange(61)
        output = smooth_root_pose(qpos)
        self.assertLess(np.abs(np.diff(output[:, 0], n=3)).mean(), np.abs(np.diff(qpos[:, 0], n=3)).mean())
        np.testing.assert_array_equal(output[:, 7:], qpos[:, 7:])
        np.testing.assert_allclose(np.linalg.norm(output[:, 3:7], axis=1), 1.0, atol=1e-12)

    # <modify>+Verify that tracking-prior fusion cannot change the root or either leg.
    def test_upper_body_blend(self):
        qpos = np.zeros((5, 36))
        reference = np.ones((5, 36))
        output = blend_upper_body(qpos, reference, strength=0.75)
        np.testing.assert_array_equal(output[:, :19], qpos[:, :19])
        np.testing.assert_allclose(output[:, 19:], 0.75)


if __name__ == "__main__":
    unittest.main()
