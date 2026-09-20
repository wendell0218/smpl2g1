import json
import unittest
from pathlib import Path

import numpy as np
import torch

from smpl2g1.contact_refinement import frozen_contact_mask, refine_contact_motion
from smpl2g1.kinematics import G1Kinematics


class KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.kinematics = G1Kinematics().double()

    def test_neutral_sole_points(self):
        qpos = torch.zeros(1, 36, dtype=torch.float64)
        qpos[:, 3] = 1
        state = self.kinematics.forward_kinematics(qpos)
        points, radii = self.kinematics.get_sole_proxy_points(state["body_pos_w"], state["body_quat_w"])
        expected = torch.tensor([
            [-0.0500023249, 0.1435064564, -0.7868637536],
            [-0.0500023249, 0.0935064556, -0.7868637536],
            [0.1199976732, 0.1485064553, -0.7868637536],
            [0.1199976732, 0.0885064567, -0.7868637536],
            [-0.0500023249, -0.0935064556, -0.7868637536],
            [-0.0500023249, -0.1435064564, -0.7868637536],
            [0.1199976732, -0.0885064567, -0.7868637536],
            [0.1199976732, -0.1485064553, -0.7868637536],
        ], dtype=torch.float64)
        torch.testing.assert_close(points[0], expected, atol=1e-9, rtol=0)
        torch.testing.assert_close(radii, torch.full((8,), 0.005, dtype=torch.float64), atol=1e-9, rtol=0)

    def test_quaternion_sign_and_batch_shape(self):
        qpos = torch.zeros(2, 3, 36, dtype=torch.float64)
        qpos[..., 3] = 1
        negative = qpos.clone()
        negative[..., 3:7] *= -1
        first = self.kinematics.forward_kinematics(qpos)
        second = self.kinematics.forward_kinematics(negative)
        self.assertEqual(first["body_pos_w"].shape, (2, 3, 30, 3))
        torch.testing.assert_close(first["body_pos_w"], second["body_pos_w"])
        torch.testing.assert_close(first["body_rot_w"], second["body_rot_w"])

    def test_sole_gradient(self):
        qpos = torch.zeros(1, 20, 36, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            qpos[..., 3] = 1
        state = self.kinematics.forward_kinematics(qpos)
        points, _ = self.kinematics.get_sole_proxy_points(state["body_pos_w"], state["body_quat_w"])
        points.square().sum().backward()
        self.assertTrue(torch.isfinite(qpos.grad).all())
        self.assertGreater(float(qpos.grad[..., :19].abs().sum()), 0)

    def test_contact_run_filter(self):
        sole = torch.zeros(8, 8, 3)
        radii = torch.full((8,), 0.005)
        sole[..., 2] = 1
        sole[1:4, :, 2] = 0.005
        valid = torch.ones(8, dtype=torch.bool)
        contact, _ = frozen_contact_mask(sole, radii, valid)
        self.assertTrue(contact[1:4].all())
        self.assertFalse(contact[[0, 4, 5, 6, 7]].any())

    def test_no_contact_returns_feasible_input(self):
        qpos = np.zeros((16, 36), dtype=np.float64)
        qpos[:, 2] = 10
        qpos[:, 3] = 1
        config = json.loads((Path(__file__).parents[1] / "smpl2g1/configs/g1_limits.json").read_text())
        limits = np.column_stack([config["lower"], config["upper"]]).astype(float)
        velocity = np.asarray(config["velocity"], dtype=float)
        output, report = refine_contact_motion(qpos, np.ones(16, dtype=bool), self.kinematics, limits, velocity)
        np.testing.assert_allclose(output, qpos)
        self.assertEqual(report["active_contact_region_frames"], 0)
        self.assertIsNone(report["best_objective"])

    def test_short_motion_is_rejected(self):
        qpos = np.zeros((15, 36), dtype=np.float64)
        qpos[:, 3] = 1
        config = json.loads((Path(__file__).parents[1] / "smpl2g1/configs/g1_limits.json").read_text())
        limits = np.column_stack([config["lower"], config["upper"]]).astype(float)
        velocity = np.asarray(config["velocity"], dtype=float)
        with self.assertRaisesRegex(ValueError, "at least 16 frames"):
            refine_contact_motion(qpos, np.ones(15, dtype=bool), self.kinematics, limits, velocity)


if __name__ == "__main__":
    unittest.main()
