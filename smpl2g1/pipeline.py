import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from .motion import build_human_frames, load_motion
from .solver import IKSolver


JOINT_NAMES = np.asarray(
    [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
)
WRIST_COLUMNS = np.asarray([26, 27, 28, 33, 34, 35])
WRIST_JOINTS = np.asarray([19, 20, 21, 26, 27, 28])


def valid_runs(valid):
    padded = np.r_[False, np.asarray(valid, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def filter_qpos(qpos, fps, limits):
    output = qpos.copy()
    if len(output) > 24:
        root_filter = butter(4, 3.0, btype="low", fs=fps, output="sos")
        rotation_filter = butter(4, 3.5, btype="low", fs=fps, output="sos")
        joint_filter = butter(4, 3.2, btype="low", fs=fps, output="sos")
        wrist_filter = butter(4, 2.8, btype="low", fs=fps, output="sos")
        output[:, :3] = sosfiltfilt(root_filter, output[:, :3], axis=0)
        quaternion = output[:, 3:7].copy()
        for index in range(1, len(quaternion)):
            if np.dot(quaternion[index - 1], quaternion[index]) < 0:
                quaternion[index] *= -1
        quaternion = sosfiltfilt(rotation_filter, quaternion, axis=0)
        output[:, 3:7] = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
        for joint_index, name in enumerate(JOINT_NAMES):
            selected_filter = wrist_filter if "wrist" in name else joint_filter
            output[:, 7 + joint_index] = sosfiltfilt(selected_filter, output[:, 7 + joint_index])
    margin = 0.005 * np.ptp(limits, axis=1)
    output[:, 7:] = np.clip(output[:, 7:], limits[:, 0] + margin, limits[:, 1] - margin)
    return output


def longest_true_run(values):
    padded = np.r_[False, values, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0


def metrics(errors, bodies, qpos, limits, fps):
    groups = {
        "non_arm": [index for index, name in enumerate(bodies) if not any(part in name for part in ("shoulder", "elbow", "wrist"))],
        "left_arm": [index for index, name in enumerate(bodies) if name.startswith("left_") and any(part in name for part in ("shoulder", "elbow", "wrist"))],
        "right_arm": [index for index, name in enumerate(bodies) if name.startswith("right_") and any(part in name for part in ("shoulder", "elbow", "wrist"))],
    }
    result = {}
    for name, indices in groups.items():
        selected = errors[:, indices]
        position = np.linalg.norm(selected[:, :, :3], axis=2)
        rotation = np.linalg.norm(selected[:, :, 3:], axis=2)
        result[name] = {
            "position_rmse_cm": float(100 * np.sqrt(np.mean(position ** 2))),
            "rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rotation ** 2)))),
        }
    margin = 0.01 * np.ptp(limits, axis=1)
    joints = qpos[:, 7:]
    near = np.logical_or(joints - limits[:, 0] <= margin, limits[:, 1] - joints <= margin)
    velocity = np.diff(qpos[:, WRIST_COLUMNS], axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    result["near_limit_fraction"] = float(near.mean())
    result["max_joint_near_limit_fraction"] = float(near.mean(axis=0).max())
    result["max_joint_near_limit_seconds"] = float(max(longest_true_run(near[:, index]) for index in range(near.shape[1])) / fps)
    result["wrist_velocity_p99_rad_s"] = float(np.quantile(np.abs(velocity), 0.99)) if velocity.size else 0.0
    result["wrist_acceleration_p99_rad_s2"] = float(np.quantile(np.abs(acceleration), 0.99)) if acceleration.size else 0.0
    result["score"] = float(
        result["non_arm"]["position_rmse_cm"] / 10
        + result["non_arm"]["rotation_rmse_deg"] / 40
        + result["left_arm"]["position_rmse_cm"] / 10
        + result["right_arm"]["position_rmse_cm"] / 10
        + result["left_arm"]["rotation_rmse_deg"] / 40
        + result["right_arm"]["rotation_rmse_deg"] / 40
        + 20 * result["max_joint_near_limit_fraction"]
        + 2 * result["max_joint_near_limit_seconds"]
    )
    return result


def arm_path(candidates, errors, bodies, limits, side):
    task_indices = [
        index
        for index, name in enumerate(bodies)
        if name.startswith(f"{side}_") and any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    joint_slice = slice(15, 22) if side == "left" else slice(22, 29)
    qpos_slice = slice(7 + joint_slice.start, 7 + joint_slice.stop)
    frame_count = len(candidates[0])
    candidate_count = len(candidates)
    unary = np.empty((frame_count, candidate_count))
    lower = limits[joint_slice, 0]
    upper = limits[joint_slice, 1]
    margin = 0.03 * (upper - lower)
    for candidate_index, (candidate, error) in enumerate(zip(candidates, errors)):
        selected = error[:, task_indices]
        fidelity = np.mean(
            np.linalg.norm(selected[:, :, :3], axis=2) / 0.1
            + np.linalg.norm(selected[:, :, 3:], axis=2) / 0.7,
            axis=1,
        )
        joints = candidate[:, qpos_slice]
        distance = np.minimum(joints - lower, upper - joints)
        barrier = np.mean(np.maximum(0.0, margin - distance) / margin, axis=1)
        unary[:, candidate_index] = fidelity + 4 * barrier
    cost = unary[0].copy()
    back = np.zeros((frame_count, candidate_count), dtype=int)
    for frame_index in range(1, frame_count):
        updated = np.empty(candidate_count)
        for candidate_index in range(candidate_count):
            current = candidates[candidate_index][frame_index, qpos_slice]
            transition = np.empty(candidate_count)
            for previous_index in range(candidate_count):
                previous = candidates[previous_index][frame_index - 1, qpos_slice]
                transition[previous_index] = (
                    cost[previous_index]
                    + 0.02 * np.sum((current - previous) ** 2)
                    + 0.15 * (candidate_index != previous_index)
                )
            best = int(np.argmin(transition))
            updated[candidate_index] = unary[frame_index, candidate_index] + transition[best]
            back[frame_index, candidate_index] = best
        cost = updated
    path = np.empty(frame_count, dtype=int)
    path[-1] = int(np.argmin(cost))
    for frame_index in range(frame_count - 1, 0, -1):
        path[frame_index - 1] = back[frame_index, path[frame_index]]
    return path


def adaptive_wrist_filter(qpos, fps, limits):
    velocity = np.diff(qpos[:, WRIST_COLUMNS], axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    velocity_p99 = float(np.quantile(np.abs(velocity), 0.99)) if velocity.size else 0.0
    acceleration_p99 = float(np.quantile(np.abs(acceleration), 0.99)) if acceleration.size else 0.0
    if velocity_p99 <= 6.75 and acceleration_p99 <= 72:
        return qpos, None
    for cutoff in (2.0, 1.5, 1.1, 1.0):
        if len(qpos) <= 24:
            break
        candidate = qpos.copy()
        selected_filter = butter(4, cutoff, btype="low", fs=fps, output="sos")
        candidate[:, WRIST_COLUMNS] = sosfiltfilt(selected_filter, candidate[:, WRIST_COLUMNS], axis=0)
        margin = 0.005 * np.ptp(limits[WRIST_JOINTS], axis=1)
        candidate[:, WRIST_COLUMNS] = np.clip(
            candidate[:, WRIST_COLUMNS],
            limits[WRIST_JOINTS, 0] + margin,
            limits[WRIST_JOINTS, 1] - margin,
        )
        velocity = np.diff(candidate[:, WRIST_COLUMNS], axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        if np.quantile(np.abs(velocity), 0.99) <= 6.75 and np.quantile(np.abs(acceleration), 0.99) <= 72:
            return candidate, cutoff
    return qpos, None


def fill_invalid(qpos, valid):
    valid_indices = np.flatnonzero(valid)
    for frame_index in np.flatnonzero(~valid):
        left = valid_indices[valid_indices < frame_index]
        right = valid_indices[valid_indices > frame_index]
        if len(left) == 0:
            qpos[frame_index] = qpos[right[0]]
        elif len(right) == 0:
            qpos[frame_index] = qpos[left[-1]]
        else:
            start, end = left[-1], right[0]
            alpha = (frame_index - start) / (end - start)
            qpos[frame_index, :3] = (1 - alpha) * qpos[start, :3] + alpha * qpos[end, :3]
            qpos[frame_index, 7:] = (1 - alpha) * qpos[start, 7:] + alpha * qpos[end, 7:]
            qpos[frame_index, 3:7] = Slerp(
                [start, end],
                Rotation.from_quat(qpos[[start, end], 3:7], scalar_first=True),
            )([frame_index]).as_quat(scalar_first=True)[0]
    return qpos


def ground_offset(model, qpos, valid, target_height):
    data = mujoco.MjData(model)
    geom_ids = [
        index
        for index in range(model.ngeom)
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    bottoms = []
    for frame in qpos[valid]:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        lowest = np.inf
        for geom_id in geom_ids:
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            center_local = model.geom_aabb[geom_id, :3]
            half_size = model.geom_aabb[geom_id, 3:]
            center_world = data.geom_xpos[geom_id] + rotation @ center_local
            lowest = min(lowest, center_world[2] - np.abs(rotation[2]) @ half_size)
        bottoms.append(lowest)
    return float(target_height - np.quantile(bottoms, 0.01))


def retarget_motion(
    input_path,
    output_path,
    body_model_dir,
    target_fps=30.0,
    up_axis="z",
    valid_path=None,
    ground_height=0.005,
):
    if target_fps <= 8:
        raise ValueError("Target FPS must be greater than 8")
    package_root = Path(__file__).resolve().parent
    xml_path = package_root / "assets/g1.xml"
    config_path = package_root / "configs/standard.json"
    motion = load_motion(input_path, target_fps, valid_path)
    frames, human_height = build_human_frames(motion, body_model_dir, up_axis)
    valid = motion["valid"]
    if not valid.any():
        raise ValueError("Valid mask has no usable frames")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    limits = model.jnt_range[1:].copy()
    qpos = np.full((len(frames), 36), np.nan, dtype=np.float64)
    run_reports = []
    for start, end in valid_runs(valid):
        run_frames = frames[start:end]
        candidates = []
        candidate_names = []
        for robust in (False, True):
            for reverse in (False, True):
                solver = IKSolver(xml_path, config_path, human_height, robust=robust)
                candidate = filter_qpos(solver.run(run_frames, reverse=reverse), target_fps, limits)
                candidates.append(candidate)
                candidate_names.append(f"{'robust' if robust else 'standard'}_{'reverse' if reverse else 'forward'}")
        evaluator = IKSolver(xml_path, config_path, human_height)
        candidate_errors = []
        candidate_metrics = []
        bodies = None
        for candidate in candidates:
            error, bodies = evaluator.errors(run_frames, candidate)
            candidate_errors.append(error)
            candidate_metrics.append(metrics(error, bodies, candidate, limits, target_fps))
        minimum_non_arm = min(item["non_arm"]["position_rmse_cm"] for item in candidate_metrics)
        eligible = [
            index
            for index, item in enumerate(candidate_metrics)
            if item["non_arm"]["position_rmse_cm"] <= max(2.0, 1.25 * minimum_non_arm)
        ]
        base_index = min(eligible, key=lambda index: candidate_metrics[index]["score"])
        selected = candidates[base_index].copy()
        path_counts = {}
        for side, columns in (("left", slice(22, 29)), ("right", slice(29, 36))):
            path = arm_path(candidates, candidate_errors, bodies, limits, side)
            for frame_index, candidate_index in enumerate(path):
                selected[frame_index, columns] = candidates[candidate_index][frame_index, columns]
            path_counts[side] = {
                name: int(np.sum(path == index))
                for index, name in enumerate(candidate_names)
            }
        selected = filter_qpos(selected, target_fps, limits)
        selected, wrist_cutoff = adaptive_wrist_filter(selected, target_fps, limits)
        selected_error, _ = evaluator.errors(run_frames, selected)
        selected_metrics = metrics(selected_error, bodies, selected, limits, target_fps)
        if selected_metrics["score"] > 1.05 * candidate_metrics[base_index]["score"]:
            selected = candidates[base_index]
            selected_metrics = candidate_metrics[base_index]
            path_counts = {}
            wrist_cutoff = None
        qpos[start:end] = selected
        run_reports.append(
            {
                "start": start,
                "end": end,
                "base": candidate_names[base_index],
                "arm_paths": path_counts,
                "wrist_lowpass_hz": wrist_cutoff,
                "candidates": {
                    name: result
                    for name, result in zip(candidate_names, candidate_metrics)
                },
                "selected": selected_metrics,
            }
        )
    qpos = fill_invalid(qpos, valid)
    vertical_offset = ground_offset(model, qpos, valid, ground_height)
    qpos[:, 2] += vertical_offset
    if not np.isfinite(qpos).all():
        raise RuntimeError("Retargeting produced non-finite values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        qpos_36=qpos.astype(np.float32),
        fps=np.float32(target_fps),
        valid=valid,
        joint_names=JOINT_NAMES,
    )
    report = {
        "pipeline": "smpl2g1-0.1.0",
        "frames": len(qpos),
        "valid_frames": int(valid.sum()),
        "fps": float(target_fps),
        "up_axis": up_axis,
        "ground_offset_m": vertical_offset,
        "runs": run_reports,
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
