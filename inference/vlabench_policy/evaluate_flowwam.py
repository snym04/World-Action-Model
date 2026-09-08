"""Run FlowWAM with VLABench's official Track-1 evaluator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_official_evaluator(repo_root):
    """Execute the unchanged official Track-1 module without optional VLA imports.

    VLABench.evaluation.__init__ eagerly imports OpenVLA/VLM dependencies.
    base.py uses absolute core imports and does not require those registries.
    No evaluator methods, task configuration or scoring logic are replaced.
    """
    source = Path(repo_root) / "VLABench/evaluation/evaluator/base.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing official evaluator: {source}")
    spec = importlib.util.spec_from_file_location(
        "_flowwam_vlabench_official_evaluator", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evaluator


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlabench-root", type=Path, required=True)
    parser.add_argument(
        "--track", default="track_1_in_distribution",
        choices=["track_1_in_distribution"],
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--visualization", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = args.vlabench_root.resolve()
    package_root = repo_root / "VLABench"
    if not (package_root / "configs" / "evaluation" / "tracks").is_dir():
        raise FileNotFoundError(f"Invalid VLABench checkout: {repo_root}")
    sys.path.insert(0, str(repo_root))
    os.environ["VLABENCH_ROOT"] = str(package_root)
    os.environ.setdefault("MUJOCO_GL", "egl")

    Evaluator = _load_official_evaluator(repo_root)
    from vlabench_policy import FlowWAMVLABenchPolicy

    track_path = (
        package_root / "configs" / "evaluation" / "tracks" /
        f"{args.track}.json"
    )
    episode_config = json.loads(track_path.read_text(encoding="utf-8"))
    tasks = list(episode_config)
    if args.tasks:
        unknown = sorted(set(args.tasks) - set(tasks))
        if unknown:
            raise ValueError(f"Tasks not in {args.track}: {unknown}")
        tasks = list(args.tasks)
    if not 1 <= args.n_episodes <= min(len(episode_config[t]) for t in tasks):
        raise ValueError("n-episodes exceeds the official track configuration")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    evaluator_source = package_root / "evaluation/evaluator/base.py"
    (args.save_dir / "evaluator_provenance.json").write_text(
        json.dumps({
            "source": str(evaluator_source),
            "sha256": hashlib.sha256(evaluator_source.read_bytes()).hexdigest(),
            "track": str(track_path),
            "track_sha256": hashlib.sha256(track_path.read_bytes()).hexdigest(),
            "loader": "unchanged official base.py via spec_from_file_location",
        }, indent=2), encoding="utf-8"
    )
    policy = FlowWAMVLABenchPolicy(
        host=args.host, port=args.port, replan_steps=args.replan_steps
    )
    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episodes,
        episode_config=episode_config,
        max_substeps=1,
        save_dir=str(args.save_dir),
        visulization=args.visualization,
        metrics=["success_rate", "intention_score", "progress_score"],
    )
    try:
        result = evaluator.evaluate(policy)
    finally:
        policy.close()
    result_path = args.save_dir / "evaluation_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
