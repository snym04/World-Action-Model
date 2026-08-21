# FlowWAM × VLABench baseline

本分支把 FlowWAM 的原始 RoboTwin 训练逻辑接到 VLABench 官方 primitive 数据与 Track-1 评测。修改只发生在 benchmark 边界；Wan2.2 双流 world model、robot-only RAFT、IDM action expert、loss、optimizer、时序和 checkpoint/resume 逻辑保持不变。

## 固定协议

- FlowWAM parent：`68abaa2b4c609febcc7b230cf06407880e85f5b8`
- VLABench source：`cf588fe60c0c7282174fe979f5913170cfe69017`
- 数据：VLABench 官方 10-task primitive HDF5，每任务 500 episodes。
- 视觉：HDF5 官方顺序为 `right, left, front, wrist`；模型输入使用 `front, left, wrist`，按 FlowWAM 原始 T-shape 拼接。
- 动作：机器人基座坐标系下 `xyz + Euler xyz + binary gripper`，共 7 维；夹爪阈值默认 `0.03`。
- 时序：33 action steps、9 video frames、visual stride 4，即 1 个 anchor + 32 个未来动作。
- 光流：默认 `robot_only + RAFT-large`；VLABench `robot_mask == 0` 表示机械臂像素。
- 优化：FlowWAM 默认 LR `1e-4`、5 epochs、action/flow loss 权重 `1.0/0.1`、30 层 IDM、action SNR shift `5.0`。
- 评测：官方 `track_1_in_distribution`，50 episodes/task；每执行 4 步重新规划，主指标 success rate，同时保存 intention/progress。

## 文件关系

```text
VLABench raw HDF5
  -> training/dataset_action_vlabench.py
  -> training/flow_action_train.py --dataset_type vlabench
  -> training/train_vlabench.sh
  -> checkpoint + action_norm_stats.npz
  -> inference/start_vlabench_server.sh
  -> inference/flow_action_server.py
  -> inference/vlabench_policy/vlabench_policy.py
  -> inference/vlabench_policy/evaluate_flowwam.py
  -> VLABench official Evaluator / Track-1 metrics and videos
```

`requirements-vlabench.lock` 固定已在 H20 验证的 Python 依赖。CUDA 包单独使用 `torch==2.5.1` 与 `torchvision==0.20.1`；运行时需确认导入版本显示 CUDA 12.4 且 RAFT 能在目标 GPU 上前向。

## 数据目录

训练数据根目录必须直接包含以下任务子目录，每个目录下可递归查找 `episode*.hdf5`：

```text
add_condiment/       insert_flower/          select_book/
select_chemistry_tube/  select_drink/        select_fruit/
select_mahjong/      select_painting/         select_poker/
select_toy/
```

每个文件需包含 `data/<episode>/observation/{rgb,robot_mask}`、`trajectory` 和 `instruction`。适配器会检查 33/9/4 的时间跨度并对 episode 尾部做右侧 padding，与 FlowWAM 原始 tail-padded 采样一致。

## 训练

正式运行必须由 WAM manifest 绑定不可变 commit、数据版本、主机、GPU、seed 和输出目录，并在 launch 前通过 SwanLab 凭据门禁。脚本所需环境变量：

```bash
source ~/.config/wam/secrets.env
PYTHON=/home/zmh/WAM/envs/flowwam-vlabench/bin/python \
MODEL_CACHE_ROOT=/home/zmh/WAM/cache/models \
DATASET_BASE_PATH=/home/zmh/WAM/data/vlabench/primitive \
OUTPUT_PATH=/home/zmh/WAM/runs/<run_id>/outputs \
NUM_GPUS=8 \
bash training/train_vlabench.sh
```

H20 的根目录替换为 `/mnt/data/zmh/WAM`。脚本会验证 Wan2.2 DiT/T5/VAE、Wan2.1 tokenizer 和 RAFT 权重已在共享 cache 中存在，不会在 H100 上隐式联网下载。

## 评测

先在模型环境启动服务器：

```bash
PYTHON=/home/zmh/WAM/envs/flowwam-vlabench/bin/python \
LOCAL_MODEL_PATH=/home/zmh/WAM/cache/models \
CHECKPOINT=/home/zmh/WAM/runs/<run_id>/outputs/step-XXXX.safetensors \
bash inference/start_vlabench_server.sh
```

再在 VLABench 环境运行官方 Track-1。正式结果默认 50 episodes/task；开发阶段可显式减小 `--n-episodes`，但不得把它标成正式 success rate：

```bash
/mnt/data/zmh/WAM/envs/vlabench/bin/python \
inference/vlabench_policy/evaluate_flowwam.py \
  --vlabench-root /mnt/data/zmh/WAM/cache/sources/VLABench \
  --n-episodes 50 \
  --replan-steps 4 \
  --visualization \
  --save-dir /mnt/data/zmh/WAM/runs/<eval_run_id>/outputs
```

## 验证

轻量 contract 测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_vlabench_contract.py
```

在扩大训练前还必须依次通过：真实 HDF5 + RAFT 数据读取、模型前后向、optimizer step、SwanLab 在线记录、checkpoint 保存、exact resume、生成 RGB/flow 视频检查，以及 VLABench 无头渲染评测 smoke。
