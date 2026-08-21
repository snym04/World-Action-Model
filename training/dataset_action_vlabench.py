"""VLABench raw-HDF5 adapter for FlowWAM training."""
from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image

from dataset_action_robotwin import ActionNormStats, WRIST_MAG_RATIO, tshape_tile
from flow_prefix_utils import (
    process_camera_flow,
    process_camera_flow_full_scene,
    tile_flow_t_shape,
    tile_flow_with_white_wrist,
)
from reversible_flow_codec import FlowCodec


VLABENCH_PRIMITIVE_TASKS = [
    "add_condiment", "insert_flower", "select_book",
    "select_chemistry_tube", "select_drink", "select_fruit",
    "select_mahjong", "select_painting", "select_poker", "select_toy",
]
# Official observation order: right, left, front, wrist.
VLABENCH_CAMERA_INDEX = {"right": 0, "left": 1, "front": 2, "wrist": 3}
VLABENCH_CAMERA_PREFIX = (
    "A multi-view video of a Franka robot in T-shape layout: "
    "the top row shows the full-size front camera view, "
    "the bottom-left shows the half-size left camera view, "
    "and the bottom-right shows the half-size wrist camera view. "
    "The robot is performing the following task: "
)


def _decode_scalar_text(value) -> str:
    arr = np.asarray(value)
    item = arr.reshape(-1)[0] if arr.ndim else arr.item()
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8")
    return str(item)


def _trajectory_to_actions(
    trajectory: np.ndarray, gripper_open_threshold: float = 0.03
) -> np.ndarray:
    """Map 8D base-frame EE waypoints to xyz+Euler+binary-open (7D)."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] < 8:
        raise ValueError(f"Expected trajectory (T, >=8), got {trajectory.shape}")
    gripper_open = (trajectory[:, -1] > gripper_open_threshold).astype(np.float32)
    return np.concatenate([trajectory[:, :6], gripper_open[:, None]], axis=1)


def _discover_records(
    data_root: str, task_names: Optional[Sequence[str]] = None
) -> List[Dict[str, str]]:
    tasks = list(task_names) if task_names else list(VLABENCH_PRIMITIVE_TASKS)
    records: List[Dict[str, str]] = []
    for task in sorted(tasks):
        pattern = os.path.join(data_root, task, "**", "episode*.hdf5")
        for hdf5_path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with h5py.File(hdf5_path, "r") as handle:
                    if "data" not in handle:
                        continue
                    for group_name in sorted(handle["data"].keys()):
                        group = handle["data"][group_name]
                        if "trajectory" in group and "observation" in group:
                            records.append({
                                "task": task,
                                "hdf5_path": hdf5_path,
                                "group_name": group_name,
                            })
            except OSError as exc:
                print(f"[VLABenchActionFlow] skip {hdf5_path}: {exc}")
    return records


def compute_vlabench_action_norm_stats(
    data_root: str,
    task_names: Optional[Sequence[str]] = None,
    gripper_open_threshold: float = 0.03,
) -> ActionNormStats:
    records = _discover_records(data_root, task_names)
    if not records:
        raise ValueError(f"No VLABench episodes under {data_root}")
    action_arrays = []
    for record in records:
        with h5py.File(record["hdf5_path"], "r") as handle:
            trajectory = handle["data"][record["group_name"]]["trajectory"][()]
            action_arrays.append(
                _trajectory_to_actions(trajectory, gripper_open_threshold)
            )
    actions = np.concatenate(action_arrays, axis=0)
    stats = ActionNormStats(
        mean=actions.mean(axis=0).astype(np.float32),
        std=actions.std(axis=0).astype(np.float32),
    )
    print(
        f"[VLABenchActionNorm] episodes={len(records)}, "
        f"steps={len(actions)}, action_dim={actions.shape[1]}"
    )
    return stats


class VLABenchActionFlowDataset:
    """Reuse FlowWAM's RGB+flow+action path with VLABench raw HDF5."""

    CAMERA_PREFIX = VLABENCH_CAMERA_PREFIX

    def __init__(
        self,
        data_root: str,
        cameras: Optional[List[str]] = None,
        size: Tuple[int, int] = (320, 256),
        num_frames: int = 33,
        visual_stride: int = 4,
        num_video_frames: Optional[int] = 9,
        task_names: Optional[List[str]] = None,
        flow_method: str = "raft",
        flow_device: str = "cuda",
        flow_max_magnitude: Optional[float] = 25.0,
        flow_mode: str = "robot_only",
        action_norm_stats: Optional[ActionNormStats] = None,
        action_norm_path: Optional[str] = None,
        gripper_open_threshold: float = 0.03,
        camera_prefix: Optional[str] = None,
    ):
        self.data_root = data_root
        self.cameras = cameras or ["front", "left", "wrist"]
        if len(self.cameras) != 3:
            raise ValueError(f"T-shape needs 3 cameras, got {self.cameras}")
        unknown = [c for c in self.cameras if c not in VLABENCH_CAMERA_INDEX]
        if unknown:
            raise ValueError(f"Unknown cameras: {unknown}")
        if flow_mode not in ("robot_only", "full_scene"):
            raise ValueError(f"Unsupported flow_mode={flow_mode}")
        if flow_method not in ("raft", "farneback"):
            raise ValueError(f"Unsupported flow_method={flow_method}")

        self.size = tuple(size)
        self.num_frames = int(num_frames)
        self.visual_stride = int(visual_stride)
        self.num_video_frames = int(
            num_video_frames if num_video_frames is not None else num_frames
        )
        span = (self.num_video_frames - 1) * self.visual_stride
        if span != self.num_frames - 1:
            raise ValueError(
                f"Temporal mismatch: video span={span}, action span={self.num_frames - 1}"
            )
        self.task_names = task_names or list(VLABENCH_PRIMITIVE_TASKS)
        self.flow_method = flow_method
        self.flow_device = flow_device
        self.flow_mode = flow_mode
        self.flow_max_magnitude = flow_max_magnitude
        self.flow_max_magnitude_wrist = (
            flow_max_magnitude * WRIST_MAG_RATIO
            if flow_max_magnitude is not None else None
        )
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.camera_prefix = camera_prefix or self.CAMERA_PREFIX
        self.codec = FlowCodec()
        self._flow_extractor = None
        self.samples = _discover_records(data_root, self.task_names)
        if not self.samples:
            raise ValueError(f"No VLABench episodes under {data_root}")

        if action_norm_stats is not None:
            self.action_norm = action_norm_stats
        elif action_norm_path and os.path.exists(action_norm_path):
            self.action_norm = ActionNormStats.load(action_norm_path)
        else:
            self.action_norm = compute_vlabench_action_norm_stats(
                data_root, self.task_names, self.gripper_open_threshold
            )
            if action_norm_path:
                os.makedirs(os.path.dirname(action_norm_path), exist_ok=True)
                self.action_norm.save(action_norm_path)
        print(
            f"[VLABenchActionFlow] episodes={len(self.samples)}, "
            f"cameras={self.cameras}, flow={self.flow_mode}/{self.flow_method}"
        )

    @property
    def flow_extractor(self):
        if self.flow_method != "raft":
            return None
        if self._flow_extractor is None:
            from raft_flow_extractor import RAFTFlowExtractor
            self._flow_extractor = RAFTFlowExtractor(device=self.flow_device)
        return self._flow_extractor

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_rgb(self, frame: np.ndarray) -> np.ndarray:
        width, height = self.size
        if frame.shape[:2] == (height, width):
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _robot_only(rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
        # VLABench defines mask=0 on robot geometry and mask=1 elsewhere.
        robot_pixels = np.asarray(robot_mask) == 0
        output = np.full_like(rgb, 255)
        output[robot_pixels] = rgb[robot_pixels]
        return output

    def _process_flow(self, rgb_native, robot_masks) -> List[Image.Image]:
        if self.flow_mode == "robot_only":
            main_robot_only = [
                self._robot_only(rgb, mask)
                for rgb, mask in zip(rgb_native[0], robot_masks[0])
            ]
            main_flow, _ = process_camera_flow(
                main_robot_only, self.size, self.codec,
                flow_method=self.flow_method,
                raft_extractor=self.flow_extractor,
                max_magnitude=self.flow_max_magnitude,
            )
            return tile_flow_with_white_wrist(main_flow, self.size)

        head_flow, _ = process_camera_flow_full_scene(
            rgb_native[0], self.size, self.codec,
            flow_method=self.flow_method,
            raft_extractor=self.flow_extractor,
            max_magnitude=self.flow_max_magnitude,
            noise_threshold=0.5,
        )
        half_size = (self.size[0] // 2, self.size[1] // 2)
        side_flow, _ = process_camera_flow_full_scene(
            rgb_native[1], half_size, self.codec,
            flow_method=self.flow_method,
            raft_extractor=self.flow_extractor,
            max_magnitude=self.flow_max_magnitude_wrist,
            noise_threshold=0.5,
        )
        wrist_flow, _ = process_camera_flow_full_scene(
            rgb_native[2], half_size, self.codec,
            flow_method=self.flow_method,
            raft_extractor=self.flow_extractor,
            max_magnitude=self.flow_max_magnitude_wrist,
            noise_threshold=0.5,
        )
        return tile_flow_t_shape(
            head_flow, side_flow, wrist_flow, target_size=self.size
        )

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        with h5py.File(sample["hdf5_path"], "r") as handle:
            group = handle["data"][sample["group_name"]]
            observation = group["observation"]
            rgb_ds = observation["rgb"]
            mask_ds = observation["robot_mask"]
            trajectory = _trajectory_to_actions(
                group["trajectory"][()], self.gripper_open_threshold
            )
            total = min(rgb_ds.shape[0], mask_ds.shape[0], len(trajectory))
            if total < 2:
                raise ValueError(f"Episode too short: {sample}")
            start = random.randint(0, total - 2)
            video_indices = [
                min(total - 1, start + i * self.visual_stride)
                for i in range(self.num_video_frames)
            ]
            action_indices = [
                min(total - 1, start + i) for i in range(self.num_frames)
            ]
            rgb_native, robot_masks = [], []
            for camera in self.cameras:
                camera_index = VLABENCH_CAMERA_INDEX[camera]
                rgb_native.append([
                    np.asarray(rgb_ds[t, camera_index], dtype=np.uint8)
                    for t in video_indices
                ])
                robot_masks.append([
                    np.asarray(mask_ds[t, camera_index]) for t in video_indices
                ])

            tiled_rgb_video = []
            for head, side, wrist in zip(*rgb_native):
                tiled_rgb_video.append(Image.fromarray(tshape_tile(
                    self._resize_rgb(head), self._resize_rgb(side),
                    self._resize_rgb(wrist)
                )))
            flow_video = self._process_flow(rgb_native, robot_masks)
            actions = self.action_norm.normalize(
                trajectory[np.asarray(action_indices, dtype=np.int64)]
            ).astype(np.float32)
            instruction = _decode_scalar_text(group["instruction"][()])

        return {
            "tiled_rgb_video": tiled_rgb_video,
            "flow_video": flow_video,
            "actions": actions,
            "video_prompt": self.camera_prefix + instruction,
            "action_prompt": instruction,
            "task": sample["task"],
        }
