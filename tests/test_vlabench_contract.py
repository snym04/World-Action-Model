import ast
import os
import random
import sys

import h5py
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, os.path.join(REPO_ROOT, "inference", "vlabench_policy"))
sys.path.insert(0, os.path.join(REPO_ROOT, "inference", "robotwin_policy"))

from dataset_action_vlabench import (  # noqa: E402
    VLABenchActionFlowDataset,
    _trajectory_to_actions,
)
from flow_action_train import (  # noqa: E402
    FlowActionTrainingModule,
    _consume_swanlab_manifest_env,
)
from vlabench_policy import FlowWAMVLABenchPolicy  # noqa: E402


def _write_episode(path, steps=40):
    rgb = np.zeros((steps, 4, 32, 32, 3), dtype=np.uint8)
    mask = np.ones((steps, 4, 32, 32), dtype=np.float32)
    for t in range(steps):
        for camera in range(4):
            rgb[t, camera] = np.array([20 * camera, 40, 100], dtype=np.uint8)
        x = min(24, 4 + t // 3)
        rgb[t, 2, 10:18, x : x + 6] = [220, 220, 220]
        mask[t, 2, 10:18, x : x + 6] = 0.0

    trajectory = np.zeros((steps, 8), dtype=np.float32)
    trajectory[:, 0] = np.linspace(0.0, 0.2, steps)
    trajectory[:, 1] = -0.1
    trajectory[:, 2] = 0.15
    trajectory[:, 6:] = np.arange(steps)[:, None] % 2 * 0.04

    with h5py.File(path, "w") as handle:
        group = handle.create_group("data").create_group("synthetic")
        group.create_dataset("instruction", data=np.asarray([b"pick the card"]))
        group.create_dataset("trajectory", data=trajectory)
        observation = group.create_group("observation")
        observation.create_dataset("rgb", data=rgb)
        observation.create_dataset("robot_mask", data=mask)


def test_vlabench_dataset_contract(tmp_path):
    task_dir = tmp_path / "select_poker"
    task_dir.mkdir()
    _write_episode(task_dir / "episode_0.hdf5")

    random.seed(0)
    dataset = VLABenchActionFlowDataset(
        data_root=str(tmp_path),
        task_names=["select_poker"],
        cameras=["front", "left", "wrist"],
        size=(32, 32),
        num_frames=5,
        num_video_frames=3,
        visual_stride=2,
        flow_method="farneback",
        flow_device="cpu",
        flow_mode="robot_only",
    )
    sample = dataset[0]
    assert len(sample["tiled_rgb_video"]) == 3
    assert len(sample["flow_video"]) == 3
    assert sample["tiled_rgb_video"][0].size == (32, 48)
    assert sample["flow_video"][0].size == (32, 48)
    assert sample["actions"].shape == (5, 7)
    assert np.isfinite(sample["actions"]).all()
    assert sample["action_prompt"] == "pick the card"


def test_action_and_policy_coordinate_contract():
    trajectory = np.array(
        [[0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 9.0, 0.04]],
        dtype=np.float32,
    )
    action = _trajectory_to_actions(trajectory)
    assert np.allclose(action[0], [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 1.0])

    robot_frame = np.array([0.2, -0.4, 0.78], dtype=np.float32)
    observation = {
        "ee_state": np.array(
            [0.3, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0, 0.04],
            dtype=np.float32,
        ),
        "robot_frame": robot_frame,
    }
    state = FlowWAMVLABenchPolicy._flowwam_state(observation, 0.03)
    assert np.allclose(state, [0.1, 0.2, 0.12, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    pos, euler, gripper = FlowWAMVLABenchPolicy._vlabench_control(
        action[0], robot_frame
    )
    assert np.allclose(pos, [0.3, -0.2, 1.08])
    assert np.allclose(euler, [0.01, 0.02, 0.03])
    assert np.allclose(gripper, [0.04, 0.04])


def test_training_module_to_keeps_pipeline_device_in_sync():
    class DummyPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.device = torch.device("meta")

    module = FlowActionTrainingModule.__new__(FlowActionTrainingModule)
    torch.nn.Module.__init__(module)
    module.pipe = DummyPipeline()

    module.to(torch.device("cpu"))

    assert module.pipe.weight.device.type == "cpu"
    assert module.pipe.device == torch.device("cpu")


def test_device_specific_seed_happens_after_accelerator_init():
    source_path = os.path.join(REPO_ROOT, "training", "flow_action_train.py")
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    launch = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_training_task"
    )
    accelerator_line = next(
        node.lineno for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Accelerator"
    )
    device_seed_line = next(
        node.lineno for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_seed"
        and any(keyword.arg == "device_specific" for keyword in node.keywords)
    )
    assert device_seed_line > accelerator_line


def test_swanlab_manifest_env_is_consumed_before_sdk_settings(monkeypatch):
    monkeypatch.setenv("SWANLAB_PROJECT", "flowwam-vlabench-idm")
    monkeypatch.setenv("SWANLAB_EXP_NAME", "manifest-bound-name")
    project, name = _consume_swanlab_manifest_env("fallback")
    assert project == "flowwam-vlabench-idm"
    assert name == "manifest-bound-name"
    assert "SWANLAB_PROJECT" not in os.environ
    assert "SWANLAB_EXP_NAME" not in os.environ


def test_full_state_is_saved_after_scheduler_step():
    source_path = os.path.join(REPO_ROOT, "training", "flow_action_train.py")
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    launch = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_training_task"
    )
    scheduler_line = next(
        node.lineno for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scheduler"
        and node.func.attr == "step"
    )
    state_save_line = next(
        node.lineno for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_save_full_state"
    )
    assert scheduler_line < state_save_line
