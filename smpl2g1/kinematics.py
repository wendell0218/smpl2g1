import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_matrix_raw(quaternion):
    w, x, y, z = torch.unbind(quaternion, -1)
    scale = 2.0 / (quaternion * quaternion).sum(-1)
    matrix = torch.stack(
        (
            1 - scale * (y * y + z * z),
            scale * (x * y - z * w),
            scale * (x * z + y * w),
            scale * (x * y + z * w),
            1 - scale * (x * x + z * z),
            scale * (y * z - x * w),
            scale * (x * z - y * w),
            scale * (y * z + x * w),
            1 - scale * (x * x + y * y),
        ),
        -1,
    )
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_to_matrix(quaternion):
    return quaternion_to_matrix_raw(F.normalize(quaternion, dim=-1))


def matrix_to_quaternion(matrix):
    shape = matrix.shape[:-2]
    values = matrix.reshape(shape + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(values, -1)
    squared_magnitude = torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), -1)
    magnitude = torch.zeros_like(squared_magnitude)
    positive = squared_magnitude > 0
    if torch.is_grad_enabled():
        magnitude[positive] = torch.sqrt(squared_magnitude[positive])
    else:
        magnitude = torch.where(positive, torch.sqrt(squared_magnitude), magnitude)
    candidates = torch.stack((
        torch.stack((magnitude[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, magnitude[..., 1] ** 2, m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, magnitude[..., 2] ** 2, m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, magnitude[..., 3] ** 2), -1),
    ), -2)
    candidates = candidates / (2 * magnitude[..., None].clamp_min(0.1))
    selected = magnitude.argmax(-1, keepdim=True).unsqueeze(-1).expand(shape + (1, 4))
    quaternion = torch.gather(candidates, -2, selected).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def axis_rotation(angle, axis, sign):
    angle = angle.squeeze(-1) * sign
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    if axis == 0:
        values = (one, zero, zero, zero, cosine, -sine, zero, sine, cosine)
    elif axis == 1:
        values = (cosine, zero, sine, zero, one, zero, -sine, zero, cosine)
    else:
        values = (cosine, -sine, zero, sine, cosine, zero, zero, zero, one)
    return torch.stack(values, -1).reshape(angle.shape + (3, 3))


def rpy_matrix(rpy):
    roll = axis_rotation(rpy[:, 0:1], 0, 1.0)
    pitch = axis_rotation(rpy[:, 1:2], 1, 1.0)
    yaw = axis_rotation(rpy[:, 2:3], 2, 1.0)
    return yaw @ pitch @ roll


class G1Kinematics(nn.Module):
    def __init__(self, config_path=None):
        super().__init__()
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "configs/g1_kinematics.json"
        spec = json.loads(Path(config_path).read_text())
        self.root_link = spec["root_link"]
        self.body_order = tuple(spec["body_order"])
        self.body_name_to_index = {name: index for index, name in enumerate(self.body_order)}
        self.parent_indices = tuple(spec["parent_body_indices"])
        self.child_indices = tuple(spec["child_body_indices"])
        axes = torch.tensor(spec["joint_axes"], dtype=torch.float32)
        axis_dimensions = torch.argmax(torch.abs(axes), dim=1)
        self.axis_dimensions = tuple(int(value) for value in axis_dimensions)
        self.axis_signs = tuple(float(axes[index, value]) for index, value in enumerate(axis_dimensions))
        origins = torch.tensor(spec["joint_origin_xyz"], dtype=torch.float32)
        origin_rpy = torch.tensor(spec["joint_origin_rpy"], dtype=torch.float32)
        self.register_buffer("joint_origin_xyz", origins, persistent=False)
        self.register_buffer("joint_origin_rotation", rpy_matrix(origin_rpy), persistent=False)
        self.register_buffer("sole_body_indices", torch.tensor(spec["sole_body_indices"], dtype=torch.long), persistent=False)
        self.register_buffer("sole_local_positions", torch.tensor(spec["sole_local_positions"], dtype=torch.float32), persistent=False)
        self.register_buffer("sole_radii", torch.tensor(spec["sole_radii"], dtype=torch.float32), persistent=False)

    def body_quat_wxyz_to_matrix(self, quaternion):
        return quaternion_to_matrix(quaternion)

    def forward_kinematics(self, qpos):
        if qpos.shape[-1] != 36:
            raise ValueError(f"Expected qpos[..., 36], got {qpos.shape}")
        root_position = qpos[..., :3]
        root_rotation = quaternion_to_matrix(qpos[..., 3:7])
        joints = qpos[..., 7:]
        batch_dimensions = qpos.ndim - 1
        positions = [None] * len(self.body_order)
        rotations = [None] * len(self.body_order)
        positions[0] = root_position
        rotations[0] = root_rotation
        origins = self.joint_origin_xyz.to(device=qpos.device, dtype=qpos.dtype)
        origin_rotations = self.joint_origin_rotation.to(device=qpos.device, dtype=qpos.dtype)
        for index, (parent, child) in enumerate(zip(self.parent_indices, self.child_indices)):
            joint_rotation = axis_rotation(joints[..., index:index + 1], self.axis_dimensions[index], self.axis_signs[index])
            origin = origins[index].view(*([1] * batch_dimensions), 3, 1)
            origin_rotation = origin_rotations[index].view(*([1] * batch_dimensions), 3, 3)
            positions[child] = positions[parent] + (rotations[parent] @ origin).squeeze(-1)
            rotations[child] = rotations[parent] @ (origin_rotation @ joint_rotation)
        body_positions = torch.stack(positions, dim=-2)
        body_rotations = torch.stack(rotations, dim=-3)
        return {
            "body_pos_w": body_positions,
            "body_rot_w": body_rotations,
            "body_quat_w": matrix_to_quaternion(body_rotations),
        }

    def get_sole_proxy_points(self, body_positions, body_quaternions):
        body_dimension = body_positions.ndim - 2
        indices = self.sole_body_indices.to(body_positions.device)
        positions = body_positions.index_select(body_dimension, indices)
        quaternions = body_quaternions.index_select(body_dimension, indices)
        rotations = quaternion_to_matrix_raw(quaternions)
        offsets = self.sole_local_positions.to(device=body_positions.device, dtype=body_positions.dtype)
        offsets = offsets.view(*([1] * (body_positions.ndim - 2)), *offsets.shape)
        points = positions + (rotations @ offsets.unsqueeze(-1)).squeeze(-1)
        radii = self.sole_radii.to(device=body_positions.device, dtype=body_positions.dtype)
        return points, radii
