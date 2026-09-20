import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .constraints import project_joint_feasibility

REGION_INDICES = ((0, 1), (2, 3), (4, 5), (6, 7))


def frozen_contact_mask(sole, radii, valid):
    bottom = sole[:, :, 2] - radii[None]
    regions = [torch.tensor(indices, device=sole.device) for indices in REGION_INDICES]
    height = torch.stack([bottom[:, region].amin(1) for region in regions], 1)
    contact = (height >= -0.02) & (height <= 0.03) & valid[:, None]
    keep = torch.zeros_like(contact)
    for region in range(4):
        edges = torch.nonzero(torch.diff(torch.cat([
            torch.zeros(1, dtype=torch.int8, device=sole.device),
            contact[:, region].to(torch.int8),
            torch.zeros(1, dtype=torch.int8, device=sole.device),
        ]))).flatten()
        for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
            if end - start >= 3:
                keep[start:end, region] = True
    return keep, regions


def refine_contact_motion(qpos, valid, kinematics, limits, velocity_limits, fps=30.0, steps=300):
    if qpos.ndim != 2 or qpos.shape[1] != 36 or valid.shape != (len(qpos),):
        raise ValueError("Contact refinement expects qpos[T,36] and valid[T]")
    if len(qpos) < 16:
        raise ValueError("Contact refinement requires at least 16 frames")
    if abs(float(fps) - 30.0) > 1e-4 or not np.isfinite(qpos).all():
        raise ValueError("Contact refinement requires finite 30 FPS motion")

    device = next(kinematics.buffers()).device
    base = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
    with torch.no_grad():
        base_fk = kinematics.forward_kinematics(base[None])
        base_sole, radii = kinematics.get_sole_proxy_points(base_fk["body_pos_w"], base_fk["body_quat_w"])
        contact, regions = frozen_contact_mask(base_sole[0], radii, valid_tensor)
        root_rotation = kinematics.body_quat_wxyz_to_matrix(base[None, :, 3:7])[0]
        base_local = torch.einsum("tji,tbj->tbi", root_rotation, base_fk["body_pos_w"][0] - base[:, None, :3])
        base_velocity = torch.diff(base_local, dim=0) * fps
        base_signal = (base_velocity[2:] + base_velocity[1:-1] + base_velocity[:-2]) / 3
        base_noise = (base_velocity[1:-1] - base_signal).square().mean().clamp_min(1e-12)
    root_rotation = root_rotation.clone()
    base_noise = base_noise.clone()

    if not contact.any():
        output, _, _ = project_joint_feasibility(
            np.asarray(qpos, dtype=np.float64), limits, 0.98 * velocity_limits / fps
        )
        report = {
            "best_step": 0,
            "initial_or_improved": "initial",
            "best_objective": None,
            "active_contact_region_frames": 0,
            "qpos_rms_change": float(np.sqrt(np.mean((output - qpos) ** 2))),
            "root_correction_filter_hz": 8.0,
            "leg_correction_filter_hz": 12.0,
        }
        return output, report

    lower = torch.as_tensor(0.975 * limits[:12, 0], dtype=torch.float32, device=device)
    upper = torch.as_tensor(0.975 * limits[:12, 1], dtype=torch.float32, device=device)
    vmax = torch.as_tensor(velocity_limits[:12], dtype=torch.float32, device=device)
    theta0 = torch.cat([base[:, :3], base[:, 7:19]], 1)
    theta = theta0.detach().clone().requires_grad_(True)
    adam_mean = torch.zeros_like(theta)
    adam_square = torch.zeros_like(theta)
    best_loss = float("inf")
    best_step = 0
    best_theta = theta.detach().clone()
    root_id = kinematics.body_name_to_index[kinematics.root_link]
    left_ids = [kinematics.body_name_to_index["left_ankle_pitch_link"], kinematics.body_name_to_index["left_ankle_roll_link"]]
    right_ids = [kinematics.body_name_to_index["right_ankle_pitch_link"], kinematics.body_name_to_index["right_ankle_roll_link"]]

    for step in range(steps + 1):
        theta.grad = None
        motion = torch.cat([theta[:, :3], base[:, 3:7], theta[:, 3:], base[:, 19:]], 1)
        fk = kinematics.forward_kinematics(motion[None])
        body = fk["body_pos_w"][0]
        sole, _ = kinematics.get_sole_proxy_points(fk["body_pos_w"], fk["body_quat_w"])
        sole = sole[0]
        region_displacement = torch.stack([
            torch.diff(sole[:, region, :2], dim=0).norm(dim=2).amax(1) for region in regions
        ], 1)
        contact_pair = contact[1:] & contact[:-1]
        skate_loss = (region_displacement[contact_pair] / 0.0033).mean()
        bottom = sole[:, :, 2] - radii[None]
        region_height = torch.stack([bottom[:, region].amin(1) for region in regions], 1)
        ground_loss = (region_height[contact] / 0.01).square().mean()
        penetration_loss = torch.relu(-bottom / 0.005).square().mean()

        delta = theta - theta0
        root_loss = (delta[:, :3] / 0.02).square().mean()
        leg_loss = (delta[:, 3:] / 0.10).square().mean()
        root_acc = (torch.diff(delta[:, :3], n=2, dim=0) / 0.001).square().mean()
        root_jerk = (torch.diff(delta[:, :3], n=3, dim=0) / 0.0005).square().mean()
        leg_vel = (torch.diff(delta[:, 3:], dim=0) / 0.02).square().mean()
        leg_acc = (torch.diff(delta[:, 3:], n=2, dim=0) / 0.01).square().mean()
        body_jerk = (torch.diff(body, n=3, dim=0) / 0.003).square().mean()

        acceleration = torch.diff(body[:, root_id], n=2, dim=0) * fps ** 2
        acceleration = torch.cat([acceleration[:, :2], torch.relu(acceleration[:, 2:3])], 1).norm(dim=1)
        acceleration = acceleration / acceleration.detach().amax().clamp_min(1e-8)
        left_speed = torch.diff(body[:, left_ids, :2], dim=0)[1:].norm(dim=2).amin(1)
        right_speed = torch.diff(body[:, right_ids, :2], dim=0)[1:].norm(dim=2).amin(1)
        pfc_loss = (left_speed * right_speed * acceleration).mean() * 10000
        local = torch.einsum("tji,tbj->tbi", root_rotation, body - theta[:, None, :3])
        velocity = torch.diff(local, dim=0) * fps
        signal = (velocity[2:] + velocity[1:-1] + velocity[:-2]) / 3
        noise_loss = (velocity[1:-1] - signal).square().mean() / base_noise
        violation = torch.relu(torch.abs(torch.diff(theta[:, 3:], dim=0)) - 0.98 * vmax[None] / fps)

        loss = (12 * skate_loss + 10 * ground_loss + 10 * penetration_loss + root_loss +
                0.3 * leg_loss + 0.5 * root_acc + root_jerk + 0.2 * leg_vel +
                2 * leg_acc + body_jerk + 5 * pfc_loss + 10 * noise_loss +
                10000 * violation.square().mean())
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_step = step
            best_theta = theta.detach().clone()
        if step == steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        with torch.no_grad():
            adam_mean.mul_(0.9).add_(theta.grad, alpha=0.1)
            adam_square.mul_(0.999).addcmul_(theta.grad, theta.grad, value=0.001)
            mean_hat = adam_mean / (1 - 0.9 ** (step + 1))
            square_hat = adam_square / (1 - 0.999 ** (step + 1))
            theta.addcdiv_(mean_hat, square_hat.sqrt().add_(1e-8), value=-0.005)
            theta[:, :3] = torch.maximum(torch.minimum(theta[:, :3], base[:, :3] + 0.12), base[:, :3] - 0.12)
            theta[:, 3:] = torch.maximum(torch.minimum(theta[:, 3:], upper[None]), lower[None])

    selected = best_theta.cpu().numpy().astype(np.float64)
    original = np.asarray(qpos, dtype=np.float64)
    correction = selected - np.column_stack([original[:, :3], original[:, 7:19]])
    root_filter = butter(4, 8.0, btype="low", fs=fps, output="sos")
    leg_filter = butter(4, 12.0, btype="low", fs=fps, output="sos")
    output = original.copy()
    output[:, :3] += sosfiltfilt(root_filter, correction[:, :3], axis=0)
    output[:, 7:19] += sosfiltfilt(leg_filter, correction[:, 3:], axis=0)
    output, _, _ = project_joint_feasibility(output, limits, 0.98 * velocity_limits / fps)
    report = {"best_step": best_step, "initial_or_improved": "initial" if best_step == 0 else "improved",
              "best_objective": best_loss, "active_contact_region_frames": int(contact.sum()),
              "qpos_rms_change": float(np.sqrt(np.mean((output - original) ** 2))),
              "root_correction_filter_hz": 8.0, "leg_correction_filter_hz": 12.0}
    return output, report
