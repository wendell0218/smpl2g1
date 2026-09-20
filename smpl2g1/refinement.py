import json
from pathlib import Path

import numpy as np

from .constraints import project_joint_feasibility


def load_g1_motion(path, joint_names):
    with np.load(path, allow_pickle=True) as data:
        key = "qpos_36" if "qpos_36" in data else "qpos"
        qpos = np.asarray(data[key], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else np.ones(len(qpos), dtype=bool)
        name_key = "joint_names" if "joint_names" in data else "robot_joint_names" if "robot_joint_names" in data else None
        names = data[name_key].astype(str).tolist() if name_key else joint_names
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 4 or valid.shape != (len(qpos),):
        raise ValueError(f"Expected qpos[T,36] with T >= 4 and valid[T], got {qpos.shape} and {valid.shape}")
    if not np.isfinite(qpos).all() or abs(fps - 30.0) > 1e-4:
        raise ValueError("Hybrid refinement requires finite 30 FPS G1 motion")
    if names != joint_names:
        raise ValueError("Input joint order does not match canonical G1 order")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-4):
        raise ValueError("Root quaternion must be normalized in wxyz order")
    return qpos, fps, valid


def smooth_root_pose(qpos, strength=0.5):
    output = qpos.copy()
    if len(output) < 3:
        return output
    quaternion = output[:, 3:7].copy()
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1
    translation = np.pad(output[:, :3], ((1, 1), (0, 0)), mode="edge")
    quaternion_pad = np.pad(quaternion, ((1, 1), (0, 0)), mode="edge")
    translation = (translation[:-2] + 2 * translation[1:-1] + translation[2:]) / 4
    quaternion_mean = (quaternion_pad[:-2] + 2 * quaternion_pad[1:-1] + quaternion_pad[2:]) / 4
    output[:, :3] = (1 - strength) * output[:, :3] + strength * translation
    quaternion = (1 - strength) * quaternion + strength * quaternion_mean
    output[:, 3:7] = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    return output


def blend_upper_body(qpos, reference, strength=0.75):
    if qpos.shape != reference.shape:
        raise ValueError(f"Upper-body reference shape {reference.shape} does not match input {qpos.shape}")
    if not 0 <= strength <= 1:
        raise ValueError("Upper-body blend strength must be in [0, 1]")
    output = qpos.copy()
    output[:, 19:] = (1 - strength) * output[:, 19:] + strength * reference[:, 19:]
    return output


def refine_motion(input_path, output_path, smpl_reference_path=None, upper_body_strength=0.75, contact=False):
    package_root = Path(__file__).resolve().parent
    config = json.loads((package_root / "configs/g1_limits.json").read_text())
    joint_names = config["joint_names"]
    limits = np.column_stack([config["lower"], config["upper"]]).astype(float)
    velocity_limits = np.asarray(config["velocity"], dtype=float)
    qpos, fps, valid = load_g1_motion(input_path, joint_names)
    original = qpos.copy()
    qpos = smooth_root_pose(qpos)
    reference_path = None
    if smpl_reference_path is not None:
        reference, reference_fps, reference_valid = load_g1_motion(smpl_reference_path, joint_names)
        if abs(reference_fps - fps) > 1e-6 or not np.array_equal(reference_valid, valid):
            raise ValueError("Upper-body reference timeline does not match the input")
        qpos = blend_upper_body(qpos, reference, upper_body_strength)
        reference_path = str(Path(smpl_reference_path).resolve())
    max_step = 0.98 * velocity_limits / fps
    qpos, lower, upper = project_joint_feasibility(qpos, limits, max_step)
    contact_report = None
    if contact:
        import torch

        from .contact_refinement import refine_contact_motion
        from .kinematics import G1Kinematics

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kinematics = G1Kinematics().to(device).eval()
        qpos, contact_report = refine_contact_motion(qpos, valid, kinematics, limits, velocity_limits, fps)
    before_position = int(((original[:, 7:] < lower) | (original[:, 7:] > upper)).any(axis=1).sum())
    before_velocity = int((np.abs(np.diff(original[:, 7:], axis=0)) > max_step).any(axis=1).sum())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, qpos_36=qpos.astype(np.float32), fps=np.float32(fps),
                        valid=valid, joint_names=np.asarray(joint_names), scene_required=np.bool_(False))
    report = {
        "pipeline": ("smpl2g1-hybrid-contact-v4" if contact else
                     "smpl2g1-rootpose-upperbody-feasibility-refinement-v3" if reference_path else
                     "smpl2g1-rootpose-feasibility-refinement-v2"),
        "input": str(Path(input_path).resolve()),
        "smpl_reference": reference_path,
        "frames": len(qpos), "fps": fps,
        "root_smoothing_strength": 0.5, "upper_body_strength": upper_body_strength if reference_path else 0.0,
        "joint_limit_scale": 0.975, "velocity_limit_fraction": 0.98,
        "position_violation_frames_before": before_position,
        "velocity_violation_intervals_before": before_velocity,
        "position_violation_frames_after": int(((qpos[:, 7:] < lower - 1e-8) | (qpos[:, 7:] > upper + 1e-8)).any(axis=1).sum()),
        "velocity_violation_intervals_after": int((np.abs(np.diff(qpos[:, 7:], axis=0)) > max_step + 1e-8).any(axis=1).sum()),
        "qpos_rms_change": float(np.sqrt(np.mean((qpos - original) ** 2))),
    }
    if contact_report is not None:
        report["contact_refinement"] = contact_report
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
