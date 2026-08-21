"""VLABench policy wrapper for a FlowWAM WebSocket inference server."""
from __future__ import annotations

import collections
import os
import sys
from typing import Deque, Dict, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOTWIN_CLIENT_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "robotwin_policy")
)
if _ROBOTWIN_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _ROBOTWIN_CLIENT_DIR)

from flow_action_client import FlowActionClient  # noqa: E402


class FlowWAMVLABenchPolicy:
    """Expose FlowWAM as VLABench's absolute end-effector policy interface."""

    name = "flowwam"
    control_mode = "ee"
    pixel_prefix = 1

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        replan_steps: int = 4,
        gripper_open_threshold: float = 0.03,
    ) -> None:
        self.client = FlowActionClient(host=host, port=port)
        self.replan_steps = int(replan_steps)
        self.gripper_open_threshold = float(gripper_open_threshold)
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")

        metadata = self.client.server_metadata
        if int(metadata.get("action_dim", -1)) != 7:
            raise ValueError(f"Expected FlowWAM action_dim=7, got {metadata}")
        if list(metadata.get("cameras", [])) != ["front", "left", "wrist"]:
            raise ValueError(f"Unexpected camera contract: {metadata}")
        if int(metadata.get("num_frames", 0)) < self.pixel_prefix + self.replan_steps:
            raise ValueError(
                "Server action horizon is shorter than anchor + replan window"
            )
        self.action_plan: Deque[np.ndarray] = collections.deque()

    @staticmethod
    def _flowwam_state(observation: Dict, threshold: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        ee_state = np.asarray(observation["ee_state"], dtype=np.float32)
        robot_frame = np.asarray(observation["robot_frame"], dtype=np.float32)
        base_position = ee_state[:3].copy() - robot_frame
        quat_wxyz = ee_state[3:7]
        quat_xyzw = [
            quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
        ]
        euler = Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(
            np.float32
        )
        gripper_open = np.float32(ee_state[-1] > threshold)
        return np.concatenate([base_position, euler, [gripper_open]]).astype(
            np.float32
        )

    @staticmethod
    def _vlabench_control(
        raw_action: np.ndarray, robot_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        target_pos = raw_action[:3].copy() + np.asarray(
            robot_frame, dtype=np.float32
        )
        target_euler = raw_action[3:6].copy()
        gripper_state = (
            np.ones(2, dtype=np.float32) * 0.04
            if raw_action[-1] >= 0.1
            else np.zeros(2, dtype=np.float32)
        )
        return target_pos, target_euler, gripper_state

    def _replan(self, observation: Dict) -> None:
        right, left, front, wrist = observation["rgb"]
        policy_input = {
            "images": {
                "front": np.asarray(front, dtype=np.uint8),
                "left": np.asarray(left, dtype=np.uint8),
                "wrist": np.asarray(wrist, dtype=np.uint8),
            },
            "instruction": str(observation["instruction"]),
            "qpos": self._flowwam_state(
                observation, self.gripper_open_threshold
            ),
        }
        actions = np.asarray(
            self.client.infer(policy_input)["actions"], dtype=np.float32
        )
        future = actions[
            self.pixel_prefix : self.pixel_prefix + self.replan_steps
        ]
        if future.shape != (self.replan_steps, 7):
            raise ValueError(f"Unexpected FlowWAM action chunk: {actions.shape}")
        self.action_plan.extend(future)

    def predict(self, observation: Dict, **kwargs):
        if not self.action_plan:
            self._replan(observation)
        raw_action = self.action_plan.popleft()
        return self._vlabench_control(raw_action, observation["robot_frame"])

    def reset(self) -> None:
        self.action_plan.clear()
        self.client.reset()

    def close(self) -> None:
        self.client.close()
