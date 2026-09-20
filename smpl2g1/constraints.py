import numpy as np


def project_joint_feasibility(qpos, limits, max_step, passes=4):
    output = qpos.copy()
    lower = np.maximum(limits[:, 0], 0.975 * limits[:, 0])
    upper = np.minimum(limits[:, 1], 0.975 * limits[:, 1])
    output[:, 7:] = np.clip(output[:, 7:], lower, upper)
    for _ in range(passes):
        for frame in range(1, len(output)):
            output[frame, 7:] = np.clip(
                output[frame, 7:],
                output[frame - 1, 7:] - max_step,
                output[frame - 1, 7:] + max_step,
            )
        for frame in range(len(output) - 2, -1, -1):
            output[frame, 7:] = np.clip(
                output[frame, 7:],
                output[frame + 1, 7:] - max_step,
                output[frame + 1, 7:] + max_step,
            )
        output[:, 7:] = np.clip(output[:, 7:], lower, upper)
    return output, lower, upper
