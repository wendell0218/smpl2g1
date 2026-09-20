import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


UMR_COMMIT = "e24fc070030dc0bb0b2c024ecb9f795a3995d725"
UMR_SLOTS_SHA256 = "55d8ff0e11b95546d339b74cbe4b9dc532dd96ce1278ba6d207c910cdb91ead9"
UMR_TEMPLATE_NAME = "smplx_neutral_betas_c4eb7419e9"
UMR_TEMPLATE_BETAS = [
    1.4774999618530273,
    0.6674000024795532,
    -1.1742000579833984,
    0.4731000065803528,
    1.2984000444412232,
    -0.2159000039100647,
    1.5276000499725342,
    -0.31520000100135803,
    -0.64410001039505,
    -0.2985999882221222,
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_motion(input_path, output_path, up_axis):
    from .motion import load_motion

    motion = load_motion(input_path, 30.0)
    if len(motion["rotations"]) < 16:
        raise ValueError("Generation requires at least 16 frames at 30 FPS")
    if not motion["valid"].all():
        raise ValueError("UMR integration currently requires every input frame to be valid")
    if motion["gender"] != "neutral":
        raise ValueError("The validated UMR correspondence currently supports neutral SMPL-X only")
    np.savez_compressed(
        output_path,
        poses=motion["rotations"].astype(np.float32),
        trans=motion["trans"].astype(np.float32),
        betas=motion["betas"].astype(np.float32),
        gender=np.asarray(motion["gender"]),
        fps=np.float32(30.0),
        valid=motion["valid"],
        output_up=np.asarray(up_axis),
    )
    return motion


def adapt_umr_result(raw_path, output_path, valid, joint_names):
    with np.load(raw_path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        names = np.asarray(data["robot_joint_names"]).astype(str).tolist()
        frame_ids = np.asarray(data["frame_ids"], dtype=int)
    if qpos.shape != (len(valid), 36) or not np.isfinite(qpos).all():
        raise ValueError(f"UMR returned invalid qpos shape or values: {qpos.shape}")
    if abs(fps - 30.0) > 1e-4 or names != list(joint_names):
        raise ValueError("UMR output does not match the canonical 30 FPS G1 format")
    if not np.array_equal(frame_ids, np.arange(len(valid))):
        raise ValueError("UMR output frame IDs do not match the canonical input timeline")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-4):
        raise ValueError("UMR returned non-normalized root quaternions")
    np.savez_compressed(
        output_path,
        qpos_36=qpos,
        fps=np.float32(30.0),
        valid=np.asarray(valid, dtype=bool),
        joint_names=np.asarray(joint_names),
        scene_required=np.bool_(False),
    )


def generate_motion(input_path, output_path, body_model_dir, umr_root, umr_env, umr_slots, up_axis="y"):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    body_model_dir = Path(body_model_dir).resolve()
    umr_root = Path(umr_root).resolve()
    umr_python = Path(umr_env).resolve() / "bin/python"
    umr_slots = Path(umr_slots).resolve()
    smplx_model_dir = body_model_dir / "smplx"
    neutral_model = smplx_model_dir / "SMPLX_NEUTRAL.npz"
    umr_script = umr_root / "scripts/retarget_smpl_to_humanoid_surface_vector_batch.py"
    umr_defaults = umr_root / "humanoid_retarget_defaults.json"
    robot_xml = umr_root / "assets/g1/g1_29dof.xml"
    required = [input_path, neutral_model, umr_python, umr_slots, umr_script, umr_defaults, robot_xml]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing generation inputs: " + ", ".join(missing))
    commit = subprocess.run(
        ["git", "-C", str(umr_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UMR_COMMIT:
        raise ValueError(f"Expected UMR commit {UMR_COMMIT}, got {commit}")
    slots_hash = sha256(umr_slots)
    if slots_hash != UMR_SLOTS_SHA256:
        raise ValueError(f"Expected validated UMR slots {UMR_SLOTS_SHA256}, got {slots_hash}")
    with np.load(umr_slots, allow_pickle=True) as slots:
        if UMR_TEMPLATE_NAME not in np.asarray(slots["names"]).astype(str).tolist():
            raise ValueError(f"UMR slots do not contain {UMR_TEMPLATE_NAME}")
        if "reconstructed_slots" not in slots:
            raise ValueError("UMR slots do not contain reconstructed_slots")

    package_root = Path(__file__).resolve().parent
    limits = json.loads((package_root / "configs/g1_limits.json").read_text())
    joint_names = limits["joint_names"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_", dir=output_path.parent) as temp:
        work = Path(temp)
        canonical_path = work / "input_30fps.npz"
        reference_path = work / "smpl2g1_reference.npz"
        raw_umr_path = work / "umr_raw.npz"
        umr_path = work / "umr_g1.npz"
        final_path = work / output_path.name
        motion = canonicalize_motion(input_path, canonical_path, up_axis)

        runtime_config = {
            "extends": str(umr_defaults),
            "smplx_model_dir": str(smplx_model_dir),
            "smpl_template": {
                "source": "manual",
                "type": "smplx",
                "use_betas": True,
                "use_gender": False,
                "gender": "neutral",
                "name": "auto",
                "betas": UMR_TEMPLATE_BETAS,
            },
            "robot": {
                "name": "unitree_g1",
                "xml": str(robot_xml),
                "point_cloud_center": "body:waist_yaw_link",
                "tpose_qpos": {
                    "left_shoulder_roll_joint": 1.57079632679,
                    "left_elbow_joint": 1.57079632679,
                    "right_shoulder_roll_joint": -1.57079632679,
                    "right_elbow_joint": 1.57079632679,
                },
                "joint_limits": {
                    "left_knee_joint": [0.1, "inf"],
                    "right_knee_joint": [0.1, "inf"],
                    "left_wrist_pitch_joint": [-0.3, 0.8],
                    "right_wrist_pitch_joint": [-0.3, 0.8],
                    "left_elbow_joint": ["-inf", 1.57079632679],
                    "right_elbow_joint": ["-inf", 1.57079632679],
                    "left_shoulder_pitch_joint": [-3.0, 1.4],
                    "right_shoulder_pitch_joint": [-3.0, 1.4],
                },
            },
            "correspondence": {
                "smpl_name": UMR_TEMPLATE_NAME,
                "slots_field": "reconstructed_slots",
            },
            "motion": {
                "data": str(canonical_path),
                "seq_key": canonical_path.stem,
                "seq_index": 0,
                "start": 0,
                "end": -1,
                "stride": 1,
                "max_frames": 0,
            },
            "solver": {
                "trajectory_warm_start_mode": "bidirectional",
                "trajectory_warm_start_bidirectional_continuity_cost": 0.1,
                "trajectory_warm_start_bidirectional_switch_cost": 0.01,
            },
            "retarget": {
                "out": str(raw_umr_path),
                "stream_chunk_frames": 300,
            },
            "view": {"enabled": False},
        }
        config_path = work / "umr_config.json"
        config_path.write_text(json.dumps(runtime_config, indent=2) + "\n")

        from .pipeline import retarget_motion

        retarget_motion(
            input_path=canonical_path,
            output_path=reference_path,
            body_model_dir=body_model_dir,
            target_fps=30.0,
            up_axis=up_axis,
        )

        command = [
            str(umr_python),
            str(umr_script),
            "--config", str(config_path),
            "--data", str(canonical_path),
            "--seq-key", canonical_path.stem,
            "--seq-index", "0",
            "--start", "0",
            "--end", "-1",
            "--stride", "1",
            "--max-frames", "0",
            "--smplx-model-dir", str(smplx_model_dir),
            "--slots", str(umr_slots),
            "--slots-field", "reconstructed_slots",
            "--smpl-name", UMR_TEMPLATE_NAME,
            "--stream-chunk-frames", "300",
            "--out", str(raw_umr_path),
        ]
        environment = os.environ.copy()
        environment["SETUPTOOLS_USE_DISTUTILS"] = "local"
        subprocess.run(command, cwd=umr_root, env=environment, check=True)
        adapt_umr_result(raw_umr_path, umr_path, motion["valid"], joint_names)

        from .refinement import refine_motion

        report = refine_motion(umr_path, final_path, reference_path)
        report.update({
            "input": str(input_path),
            "smpl_reference": "generated",
            "umr_commit": commit,
            "umr_slots_sha256": slots_hash,
            "up_axis": up_axis,
        })
        final_report = final_path.with_suffix(".json")
        final_report.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(final_path, output_path)
        os.replace(final_report, output_path.with_suffix(".json"))
    return report
