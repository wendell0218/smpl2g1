import copy
import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class IKSolver:
    def __init__(self, xml_path, config_path, human_height, robust=False):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.configuration = mink.Configuration(self.model)
        config = json.loads(Path(config_path).read_text())
        if robust:
            config = copy.deepcopy(config)
            for table_name, position_cost in (("ik_match_table1", 10), ("ik_match_table2", 20)):
                for robot_body, entry in config[table_name].items():
                    if "shoulder" in robot_body or "elbow" in robot_body:
                        entry[1], entry[2] = position_cost, 3
                    elif "wrist" in robot_body:
                        entry[1], entry[2] = position_cost, 1
        ratio = human_height / config["human_height_assumption"]
        self.scale = {name: factor * ratio for name, factor in config["human_scale_table"].items()}
        self.ground = config["ground_height"] * np.array([0.0, 0.0, 1.0])
        self.tasks = []
        self.task_bodies = []
        self.body_tasks = []
        for table_name, enabled_name in (("ik_match_table1", "use_ik_match_table1"), ("ik_match_table2", "use_ik_match_table2")):
            stage_tasks = []
            stage_bodies = []
            position_offsets = {}
            rotation_offsets = {}
            if config[enabled_name]:
                for robot_body, entry in config[table_name].items():
                    human_body, position_cost, rotation_cost, position_offset, rotation_offset = entry
                    if position_cost == 0 and rotation_cost == 0:
                        continue
                    task = mink.FrameTask(
                        frame_name=robot_body,
                        frame_type="body",
                        position_cost=position_cost,
                        orientation_cost=rotation_cost,
                        lm_damping=1,
                    )
                    stage_tasks.append(task)
                    stage_bodies.append(human_body)
                    position_offsets[human_body] = np.asarray(position_offset) - self.ground
                    rotation_offsets[human_body] = Rotation.from_quat(rotation_offset, scalar_first=True)
            self.tasks.append(stage_tasks)
            self.task_bodies.append(stage_bodies)
            self.body_tasks.append((position_offsets, rotation_offsets))
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.damping = 0.5
        self.max_iterations = 10

    def update_targets(self, frame):
        root_position = frame["pelvis"][0]
        scaled = {}
        for name, (position, quaternion) in frame.items():
            if name not in self.scale:
                continue
            if name == "pelvis":
                scaled_position = self.scale[name] * position
            else:
                scaled_position = self.scale[name] * (position - root_position) + self.scale["pelvis"] * root_position
            scaled[name] = (scaled_position, quaternion)
        for stage_index in range(2):
            position_offsets, rotation_offsets = self.body_tasks[stage_index]
            for task, human_body in zip(self.tasks[stage_index], self.task_bodies[stage_index]):
                position, quaternion = scaled[human_body]
                orientation = Rotation.from_quat(quaternion, scalar_first=True) * rotation_offsets[human_body]
                target_position = position + orientation.apply(position_offsets[human_body])
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(orientation.as_quat(scalar_first=True)),
                        target_position,
                    )
                )

    def initialize(self, frame):
        self.update_targets(frame)
        pelvis_task = None
        for stage_tasks, stage_bodies in zip(self.tasks, self.task_bodies):
            if "pelvis" in stage_bodies:
                pelvis_task = stage_tasks[stage_bodies.index("pelvis")]
                break
        target = pelvis_task.transform_target_to_world
        qpos = self.configuration.data.qpos.copy()
        qpos[:3] = target.translation()
        qpos[3:7] = target.rotation().parameters()
        self.configuration.update(qpos)

    def step(self, frame):
        self.update_targets(frame)
        for tasks in self.tasks:
            if not tasks:
                continue
            current = np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks]))
            for _ in range(self.max_iterations + 1):
                dt = self.model.opt.timestep
                velocity = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=tasks,
                    dt=dt,
                    solver="daqp",
                    damping=self.damping,
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, dt)
                updated = np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks]))
                if current - updated <= 0.001:
                    break
                current = updated
        return self.configuration.data.qpos.copy()

    def run(self, frames, reverse=False, warmup=9):
        order = list(range(len(frames) - 1, -1, -1)) if reverse else list(range(len(frames)))
        self.initialize(frames[order[0]])
        for _ in range(warmup):
            self.step(frames[order[0]])
        output = np.empty((len(frames), self.model.nq), dtype=np.float64)
        for frame_index in order:
            output[frame_index] = self.step(frames[frame_index])
        return output

    def errors(self, frames, qpos):
        stage = 1 if self.tasks[1] else 0
        errors = []
        for frame, configuration in zip(frames, qpos):
            self.update_targets(frame)
            self.configuration.update(configuration)
            errors.append([task.compute_error(self.configuration) for task in self.tasks[stage]])
        return np.asarray(errors), list(self.task_bodies[stage])
