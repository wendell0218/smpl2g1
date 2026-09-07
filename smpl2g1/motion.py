from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation, Slerp
from smplx.joint_names import JOINT_NAMES


TARGET_NAMES = (
    "pelvis",
    "spine3",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def value(data, names, required=True):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    if required:
        raise KeyError(f"Missing one of: {', '.join(names)}")
    return None


def rotations_to_rotvec(array):
    array = np.asarray(array)
    if array.shape[-2:] == (3, 3):
        return Rotation.from_matrix(array.reshape(-1, 3, 3)).as_rotvec().reshape(*array.shape[:-2], 3)
    return array.reshape(array.shape[0], -1, 3)


def resample_rotvec(rotvec, source_fps, target_fps):
    if len(rotvec) == 1 or abs(source_fps - target_fps) < 1e-8:
        return rotvec.copy()
    count = max(1, int(round((len(rotvec) - 1) * target_fps / source_fps)) + 1)
    source_time = np.arange(len(rotvec)) / source_fps
    target_time = np.minimum(np.arange(count) / target_fps, source_time[-1])
    output = np.empty((count, rotvec.shape[1], 3), dtype=np.float64)
    for joint_index in range(rotvec.shape[1]):
        output[:, joint_index] = Slerp(source_time, Rotation.from_rotvec(rotvec[:, joint_index]))(target_time).as_rotvec()
    return output


def load_motion(path, target_fps, valid_path=None):
    with np.load(path, allow_pickle=True) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    poses = value(data, ("poses",), required=False)
    if poses is not None:
        rotations = rotations_to_rotvec(poses)
        root = rotations[:, :1]
        body = rotations[:, 1:22]
    else:
        root = rotations_to_rotvec(value(data, ("global_orient", "root_orient")))[:, :1]
        body = rotations_to_rotvec(value(data, ("body_pose", "pose_body")))[:, :21]
    frames = len(root)
    if len(body) != frames:
        raise ValueError("Pose arrays have different frame counts")
    if body.shape[1] < 21:
        body = np.pad(body, ((0, 0), (0, 21 - body.shape[1]), (0, 0)))
    trans = value(data, ("trans", "transl", "translation"))
    if trans.shape != (frames, 3):
        raise ValueError("Translation must have shape [frames, 3]")
    betas = value(data, ("betas",), required=False)
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    else:
        betas = np.asarray(betas)
        betas = betas.reshape(-1, betas.shape[-1])[0]
    fps_array = value(data, ("mocap_framerate", "mocap_frame_rate", "fps"), required=False)
    source_fps = float(np.asarray(30.0 if fps_array is None else fps_array).reshape(-1)[0])
    gender_value = data.get("gender", np.asarray("neutral"))
    gender = str(np.asarray(gender_value).reshape(-1)[0]).lower()
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"
    valid = np.ones(frames, dtype=bool) if "valid" not in data else np.asarray(data["valid"], dtype=bool)
    if valid_path is not None:
        valid = np.load(valid_path, allow_pickle=False).astype(bool)
    if valid.shape != (frames,):
        raise ValueError("Valid mask must have shape [frames]")
    local_rotations = np.concatenate([root, body], axis=1)
    resampled_rotations = resample_rotvec(local_rotations, source_fps, target_fps)
    if len(resampled_rotations) == frames:
        resampled_trans = trans.astype(np.float64)
        resampled_valid = valid.copy()
    else:
        source_time = np.arange(frames) / source_fps
        target_time = np.minimum(np.arange(len(resampled_rotations)) / target_fps, source_time[-1])
        resampled_trans = np.column_stack([np.interp(target_time, source_time, trans[:, axis]) for axis in range(3)])
        source_indices = np.minimum(np.rint(target_time * source_fps).astype(int), frames - 1)
        resampled_valid = valid[source_indices]
    return {
        "rotations": resampled_rotations,
        "trans": resampled_trans,
        "betas": betas,
        "gender": gender,
        "valid": resampled_valid,
        "fps": float(target_fps),
    }


def build_human_frames(motion, body_model_dir, up_axis):
    model = smplx.create(
        str(Path(body_model_dir)),
        "smplx",
        gender=motion["gender"],
        use_pca=False,
        num_betas=min(16, len(motion["betas"])),
    )
    rotations = motion["rotations"]
    frame_count = len(rotations)
    betas = np.asarray(motion["betas"], dtype=np.float32)[: model.num_betas]
    if len(betas) < model.num_betas:
        betas = np.pad(betas, (0, model.num_betas - len(betas)))
    output = model(
        betas=torch.tensor(betas).view(1, -1),
        global_orient=torch.tensor(rotations[:, 0], dtype=torch.float32),
        body_pose=torch.tensor(rotations[:, 1:22].reshape(frame_count, 63), dtype=torch.float32),
        transl=torch.tensor(motion["trans"], dtype=torch.float32),
        left_hand_pose=torch.zeros(frame_count, 45),
        right_hand_pose=torch.zeros(frame_count, 45),
        jaw_pose=torch.zeros(frame_count, 3),
        leye_pose=torch.zeros(frame_count, 3),
        reye_pose=torch.zeros(frame_count, 3),
        return_full_pose=True,
    )
    parents = model.parents.detach().cpu().numpy().astype(int)
    local = output.full_pose.detach().cpu().numpy().reshape(frame_count, -1, 3)
    joints = output.joints.detach().cpu().numpy()[:, : len(parents)]
    global_rotation = [Rotation.from_rotvec(local[:, 0])]
    for joint_index in range(1, len(parents)):
        global_rotation.append(global_rotation[parents[joint_index]] * Rotation.from_rotvec(local[:, joint_index]))
    axis_rotation = Rotation.identity() if up_axis == "z" else Rotation.from_euler("x", 90, degrees=True)
    name_to_index = {name: index for index, name in enumerate(JOINT_NAMES[: len(parents)])}
    frames = []
    for frame_index in range(frame_count):
        frame = {}
        for name in TARGET_NAMES:
            joint_index = name_to_index[name]
            position = axis_rotation.apply(joints[frame_index, joint_index])
            orientation = axis_rotation * global_rotation[joint_index][frame_index]
            frame[name] = (position, orientation.as_quat(scalar_first=True))
        frames.append(frame)
    height = float(1.66 + 0.1 * float(betas[0]))
    return frames, height
