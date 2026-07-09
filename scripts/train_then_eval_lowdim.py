#!/usr/bin/env python3
"""Run low-dim BC training, then run the configured low-dim eval scripts."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "factr" / "train_bc_policy.py"
BASE_EVAL_CONFIG = PROJECT_ROOT / "scripts" / "eval_params.yaml"
EVAL_SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "eval_single_episode_lowdim.py"
EVAL_Z_SCRIPT = PROJECT_ROOT / "scripts" / "eval_z_distr.py"
LATEST_CHECKPOINT = "latest_ckpt.ckpt"


def _read_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _run_streamed(command, *, cwd: Path) -> Tuple[int, Optional[Path]]:
    """Run a command with live output and capture the training run directory."""
    run_dir = None
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        stripped = line.strip()
        if stripped.startswith("Run directory:"):
            run_dir = Path(stripped.split(":", 1)[1].strip()).expanduser()

    return process.wait(), run_dir


def _run_checked(command, *, cwd: Path, stage_name: str) -> None:
    print(f"\n=== {stage_name} ===", flush=True)
    result = subprocess.run([str(part) for part in command], cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"{stage_name} failed with exit code {result.returncode}.")


def _load_training_params(run_dir: Path) -> Dict:
    exp_config_path = run_dir / "exp_config.yaml"
    cfg = _read_yaml(exp_config_path)
    params = cfg.get("params", cfg)
    if not isinstance(params, dict):
        raise SystemExit(f"Could not read training params from {exp_config_path}")
    return params


def _infer_project_prefix(run_dir: Path, dataset_name: str) -> str:
    project_key = run_dir.parent.name
    if project_key == dataset_name:
        return ""
    if project_key.endswith(dataset_name):
        return project_key[: -len(dataset_name)]
    return ""


def _verify_training_outputs(run_dir: Optional[Path]) -> Path:
    if run_dir is None:
        raise SystemExit("Training finished, but no 'Run directory:' line was found in the output.")

    rollout_dir = run_dir / "rollout"
    required_paths = [
        rollout_dir / LATEST_CHECKPOINT,
        rollout_dir / "exp_config.yaml",
        rollout_dir / "rollout_config.yaml",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Training did not produce the required eval files:\n{missing_text}")

    return run_dir


def _write_eval_config(run_dir: Path) -> Path:
    train_params = _load_training_params(run_dir)
    dataset_name = str(train_params.get("dataset_name", "")).strip()
    exp_name = str(train_params.get("exp_name", run_dir.name)).strip() or run_dir.name
    if not dataset_name:
        raise SystemExit(f"Training config in {run_dir / 'exp_config.yaml'} has no dataset_name.")

    eval_cfg = _read_yaml(BASE_EVAL_CONFIG)
    shared = dict(eval_cfg.get("shared", {}))

    # Keep eval plotting settings intact, but point them at the checkpoint just trained.
    shared["dataset_name"] = dataset_name
    shared["dataset_project_prefix"] = _infer_project_prefix(run_dir, dataset_name)
    shared["buffer_set_name"] = "auto"
    shared["run_name"] = exp_name
    shared["checkpoint_name"] = LATEST_CHECKPOINT
    eval_cfg["shared"] = shared

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_eval_params.yaml",
        prefix="factr_lowdim_",
        delete=False,
    )
    with temp:
        yaml.safe_dump(eval_cfg, temp, sort_keys=False)

    temp_path = Path(temp.name)
    print(f"\nTemporary eval config: {temp_path}", flush=True)
    print(f"Eval run: checkpoints/{shared['dataset_project_prefix']}{dataset_name}/{exp_name}/rollout", flush=True)
    return temp_path


def main() -> None:
    train_cmd = [sys.executable, TRAIN_SCRIPT, "--config-name", "train_bc_lowdim"]

    print("=== Training low-dim BC/CVAE policy ===", flush=True)
    return_code, run_dir = _run_streamed(train_cmd, cwd=PROJECT_ROOT)
    if return_code != 0:
        raise SystemExit(f"Training failed with exit code {return_code}; eval scripts were not run.")

    run_dir = _verify_training_outputs(run_dir)
    eval_config = _write_eval_config(run_dir)

    _run_checked(
        [sys.executable, EVAL_SINGLE_SCRIPT, "--config", eval_config],
        cwd=PROJECT_ROOT,
        stage_name="Evaluating single low-dim episode",
    )
    _run_checked(
        [sys.executable, EVAL_Z_SCRIPT, "--config", eval_config],
        cwd=PROJECT_ROOT,
        stage_name="Evaluating low-dim Z distributions",
    )

    print("\nTrain and eval pipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
