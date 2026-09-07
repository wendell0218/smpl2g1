import unittest

import numpy as np

from smpl2g1.pipeline import adaptive_wrist_filter, valid_runs


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


if __name__ == "__main__":
    unittest.main()
