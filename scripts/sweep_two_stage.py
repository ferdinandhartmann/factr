#!/usr/bin/env python3
"""Two-stage W&B sweep launcher using a declarative Python config style.

This is intentionally based on the existing shell launcher, but follows the
article pattern more closely by constructing sweep configurations as Python
dictionaries and then creating sweeps programmatically.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_MODULE = "factr.train_bc_policy"
WANDB_CLI = [sys.executable, "-m", "wandb"]


def normalize_sweep_goal(raw_goal: str) -> str:
    goal = raw_goal.strip().lower()
    if goal in {"min", "minimize"}:
        return "minimize"
    if goal in {"max", "maximize"}:
        return "maximize"
    raise ValueError(f"Unsupported SWEEP_GOAL={raw_goal!r}. Use 'minimize' or 'maximize'.")


PROJECT = os.environ.get("WANDB_PROJECT", "aiact_sweep_4")
ENTITY = os.environ.get("WANDB_ENTITY", "")
SWEEP_METRIC = os.environ.get("SWEEP_METRIC", "eval/sweep_score")
SWEEP_GOAL = normalize_sweep_goal(os.environ.get("SWEEP_GOAL", "minimize"))

GPU_IDS = tuple(int(item.strip()) for item in os.environ.get("GPU_IDS", "0,1,2").split(",") if item.strip())
AGENTS_PER_GPU = int(os.environ.get("AGENTS_PER_GPU", "3"))
MIN_WAVES_PER_STAGE = int(os.environ.get("MIN_WAVES_PER_STAGE", "3"))
AGENT_MAX_RUNS_PER_WORKER = os.environ.get("AGENT_MAX_RUNS_PER_WORKER", "").strip()

SHORT_RUN_CAP = int(os.environ.get("SHORT_RUN_CAP", "54"))
LONG_RUN_CAP = int(os.environ.get("LONG_RUN_CAP", "27"))

RESULTS_ROOT = REPO_ROOT / "checkpoints" / "sweeps_article_style"

ACTIVE_PROCS: List[subprocess.Popen] = []


@dataclass(frozen=True)
class StageConfig:
    name: str
    sweep_name: str
    run_cap: int
    max_steps: int
    method: str


SHORT_STAGE = StageConfig(name="short_5000", sweep_name="factr-lowdim-short-5000-article-style", run_cap=SHORT_RUN_CAP, max_steps=5000, method="random")

LONG_STAGE = StageConfig(name="long_15000", sweep_name="factr-lowdim-long-15000-article-style", run_cap=LONG_RUN_CAP, max_steps=15000, method="bayes")


def require_cmd(cmd: str) -> None:
    if cmd == "wandb":
        subprocess.run(WANDB_CLI + ["--version"], cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
        return
    raise FileNotFoundError(f"Unsupported command check: {cmd}")


def resolve_run_cap(requested: int, total_agents: int) -> int:
    min_runs = total_agents * max(1, MIN_WAVES_PER_STAGE)
    return max(int(requested), int(min_runs))


def stage_dir(stage_name: str) -> Path:
    out_dir = RESULTS_ROOT / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def validate_sweep_config(stage: StageConfig, sweep_config: Dict[str, Any]) -> None:
    goal = sweep_config.get("metric", {}).get("goal")
    if goal not in {"minimize", "maximize"}:
        raise ValueError(f"Invalid sweep goal for stage={stage.name}: {goal!r}")

    for param_name, param_spec in sweep_config.get("parameters", {}).items():
        if not isinstance(param_spec, dict):
            continue

        distribution = param_spec.get("distribution")
        if distribution not in {"uniform", "log_uniform_values"}:
            continue

        min_value = param_spec.get("min")
        max_value = param_spec.get("max")
        if min_value is None or max_value is None:
            raise ValueError(f"Sweep parameter '{param_name}' in stage={stage.name} is missing min/max for {distribution}.")
        if float(min_value) >= float(max_value):
            raise ValueError(f"Sweep parameter '{param_name}' in stage={stage.name} has invalid range: " f"min={min_value} max={max_value}.")


def build_sweep_config(stage: StageConfig) -> Dict[str, Any]:
    hydra_dir = f"{REPO_ROOT}/checkpoints/sweeps/{stage.name}/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}_${{oc.env:WANDB_RUN_ID,na}}"

    sweep_config: Dict[str, Any] = {
        "program": PROGRAM_MODULE,
        "name": stage.sweep_name,
        "method": stage.method,
        "project": PROJECT,
        "metric": {"name": SWEEP_METRIC, "goal": SWEEP_GOAL},
        "run_cap": int(stage.run_cap),
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            "${program}",
            "--config-name",
            "train_bc_lowdim",
            "hydra.job.env_set.CUDA_VISIBLE_DEVICES=${oc.env:CUDA_VISIBLE_DEVICES,0}",
            f"hydra.run.dir={hydra_dir}",
            f"wandb.project={PROJECT}",
            f"wandb.group=sweep_{stage.name}",
            "${args_no_hyphens}",
        ],
        "parameters": {
            "max_iterations": {"value": int(stage.max_steps)},
            "seed": {"value": 42},
            "lr": {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.005},
            "batch_size": {"values": [64, 128]},
            "obs_window": {"values": [4, 8]},
            "train_log_freq": {"value": 50},
            "eval_freq": {"value": 250},
            "eval_freq_plot": {"value": 5000},
            "ac_chunk": {"value": 30},
            "agent.beta": {"distribution": "log_uniform_values", "min": 1.0e-03, "max": 5.0e-02},
            "agent.kl_balance_alpha": {"distribution": "uniform", "min": 0.55, "max": 0.95},
            "agent.categorical_temperature": {"distribution": "uniform", "min": 0.45, "max": 1.00},
            "agent.latent_distribution": {"values": ["categorical", "gaussian"]},
            "agent.d_z": {"values": [4, 8, 12, 16, 32]},
            "agent.categorical_num_categories": {"values": [4, 8, 16, 32]},
            "agent.categorical_num_variables": {"values": [4, 8, 16, 32]},
            "agent.token_dim": {"values": [96, 128, 160]},
            "agent.hidden_dim": {"values": [192, 256, 384]},
            "task.eval_diversity_num_samples": {"value": 10},
            "task.eval_diversity_max_batches": {"value": 2},
            "task.sweep_target_min_diversity": {"value": 0.015},
            "task.sweep_target_min_kl": {"value": 0.2},
            "task.sweep_diversity_penalty": {"value": 1.5},
            "task.sweep_kl_penalty": {"value": 0.08},
        },
    }
    if ENTITY:
        sweep_config["entity"] = ENTITY
    return sweep_config


def write_sweep_config(stage: StageConfig, sweep_config: Dict[str, Any]) -> Path:
    out_path = stage_dir(stage.name) / "sweep_config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sweep_config, f, sort_keys=False)
    return out_path


def parse_sweep_id(output: str) -> str:
    marker = "wandb agent "
    for line in output.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    raise RuntimeError("Failed to parse sweep id from wandb sweep output.")


def create_sweep(config_path: Path) -> str:
    proc = subprocess.run(WANDB_CLI + ["sweep", str(config_path)], cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"wandb sweep failed for {config_path} with exit code {proc.returncode}.")
    return parse_sweep_id(proc.stdout + "\n" + proc.stderr)


def cleanup_processes() -> None:
    global ACTIVE_PROCS
    for proc in ACTIVE_PROCS:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    if ACTIVE_PROCS:
        try:
            import time

            time.sleep(2)
        except Exception:
            pass

    for proc in ACTIVE_PROCS:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    ACTIVE_PROCS = []


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}, stopping sweep workers...", file=sys.stderr)
        cleanup_processes()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def launch_agents(stage: StageConfig, sweep_id: str) -> None:
    global ACTIVE_PROCS
    total_agents = len(GPU_IDS) * AGENTS_PER_GPU
    if AGENT_MAX_RUNS_PER_WORKER:
        per_agent_count = int(AGENT_MAX_RUNS_PER_WORKER)
    else:
        per_agent_count = int(stage.run_cap)

    print(
        f"Launching stage={stage.name} with {total_agents} agents "
        f"({AGENTS_PER_GPU}/GPU on GPUs {list(GPU_IDS)}), "
        f"up to {per_agent_count} runs/agent, sweep run_cap={stage.run_cap}."
    )

    ACTIVE_PROCS = []
    for gpu in GPU_IDS:
        for slot in range(1, AGENTS_PER_GPU + 1):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
            proc = subprocess.Popen(WANDB_CLI + ["agent", "--count", str(per_agent_count), sweep_id], cwd=str(REPO_ROOT), env=env, start_new_session=True)
            ACTIVE_PROCS.append(proc)
            print(f"  started stage={stage.name} gpu={gpu} slot={slot} pid={proc.pid}")

    rc = 0
    for proc in ACTIVE_PROCS:
        ret = proc.wait()
        if ret != 0:
            rc = ret
    ACTIVE_PROCS = []

    if rc != 0:
        raise RuntimeError(f"One or more agents failed in stage={stage.name} with exit code {rc}.")


def get_summary_metric(run: Any, metric_name: str) -> Any:
    summary_json = getattr(run.summary, "_json_dict", None)
    if isinstance(summary_json, dict):
        if metric_name in summary_json:
            return summary_json.get(metric_name)

        cursor: Any = summary_json
        for part in metric_name.split("/"):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                cursor = None
                break
        if cursor is not None:
            return cursor

    try:
        summary_dict = dict(run.summary)
    except Exception:
        summary_dict = {}

    if metric_name in summary_dict:
        return summary_dict.get(metric_name)

    cursor = summary_dict
    for part in metric_name.split("/"):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            return None
    return cursor


def is_valid_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_best_run_summary(stage: StageConfig, sweep_id: str) -> None:
    out_dir = stage_dir(stage.name)
    out_json = out_dir / "best_run.json"
    out_txt = out_dir / "best_run.txt"

    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    eligible: List[Tuple[float, Any]] = []
    for run in sweep.runs:
        metric_value = get_summary_metric(run, SWEEP_METRIC)
        if is_valid_metric(metric_value):
            eligible.append((float(metric_value), run))

    reverse = SWEEP_GOAL == "maximize"
    eligible.sort(key=lambda item: item[0], reverse=reverse)

    payload: Dict[str, Any] = {
        "stage": stage.name,
        "sweep_id": sweep_id,
        "metric": SWEEP_METRIC,
        "goal": SWEEP_GOAL,
        "total_runs": len(sweep.runs),
        "eligible_runs": len(eligible),
        "best": None,
    }

    text_lines = [f"stage={stage.name}", f"sweep_id={sweep_id}", f"metric={SWEEP_METRIC}", f"goal={SWEEP_GOAL}", f"total_runs={len(sweep.runs)}", f"eligible_runs={len(eligible)}"]

    if eligible:
        best_value, best_run = eligible[0]
        cleaned_config = {k: v for k, v in best_run.config.items() if not str(k).startswith("_")}
        run_url = getattr(best_run, "url", None)
        if not run_url:
            path = getattr(best_run, "path", None)
            if path and len(path) >= 2:
                run_url = f"https://wandb.ai/{path[0]}/{path[1]}/runs/{best_run.id}"

        payload["best"] = {"run_id": best_run.id, "run_name": best_run.name, "run_state": best_run.state, "metric_value": best_value, "run_url": run_url, "config": cleaned_config}
        text_lines.extend(
            [
                f"best_run_id={best_run.id}",
                f"best_run_name={best_run.name}",
                f"best_run_state={best_run.state}",
                f"best_metric_value={best_value}",
                f"best_run_url={run_url}",
                "best_config_json=",
                json.dumps(cleaned_config, sort_keys=True),
            ]
        )
    else:
        text_lines.append("best_run_id=none")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")

    print(f"Wrote best-run summary: {out_txt}")


def write_overall_best_summary(stages: Sequence[StageConfig]) -> None:
    candidates: List[Tuple[str, str, str, float, Dict[str, Any]]] = []
    for stage in stages:
        best_json = stage_dir(stage.name) / "best_run.json"
        if not best_json.exists():
            continue
        with open(best_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best")
        if not best:
            continue
        metric_value = best.get("metric_value")
        if not isinstance(metric_value, (int, float)):
            continue
        candidates.append((payload.get("stage", stage.name), payload.get("metric", SWEEP_METRIC), payload.get("goal", SWEEP_GOAL), float(metric_value), best))

    summary: Dict[str, Any] = {"candidates": len(candidates), "best_overall": None}
    if candidates:
        goal = candidates[0][2]
        reverse = goal == "maximize"
        candidates.sort(key=lambda item: item[3], reverse=reverse)
        stage_name, metric, goal, metric_value, best = candidates[0]
        summary["best_overall"] = {
            "stage": stage_name,
            "metric": metric,
            "goal": goal,
            "metric_value": metric_value,
            "run_id": best.get("run_id"),
            "run_name": best.get("run_name"),
            "run_url": best.get("run_url"),
            "config": best.get("config", {}),
        }

    out_json = RESULTS_ROOT / "best_overall.json"
    out_txt = RESULTS_ROOT / "best_overall.txt"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"candidates={summary['candidates']}\n")
        best = summary.get("best_overall")
        if best:
            f.write(f"stage={best['stage']}\n")
            f.write(f"metric={best['metric']}\n")
            f.write(f"goal={best['goal']}\n")
            f.write(f"metric_value={best['metric_value']}\n")
            f.write(f"run_id={best['run_id']}\n")
            f.write(f"run_name={best['run_name']}\n")
            f.write(f"run_url={best['run_url']}\n")
            f.write("config_json=\n")
            f.write(json.dumps(best.get("config", {}), sort_keys=True) + "\n")
        else:
            f.write("best_overall=none\n")

    print(f"Wrote overall best summary: {out_txt}")


def run_stage(stage: StageConfig) -> str:
    sweep_config = build_sweep_config(stage)
    validate_sweep_config(stage, sweep_config)
    config_path = write_sweep_config(stage, sweep_config)
    print(f"Creating {stage.name} sweep from {config_path}")
    sweep_id = create_sweep(config_path)
    print(f"{stage.name} sweep id: {sweep_id}")
    launch_agents(stage, sweep_id)
    write_best_run_summary(stage, sweep_id)
    return sweep_id


def main() -> None:
    require_cmd("wandb")

    total_agents = len(GPU_IDS) * AGENTS_PER_GPU
    if total_agents < 1:
        raise ValueError("Need at least one sweep agent.")

    short_stage = StageConfig(
        name=SHORT_STAGE.name,
        sweep_name=SHORT_STAGE.sweep_name,
        run_cap=resolve_run_cap(SHORT_STAGE.run_cap, total_agents),
        max_steps=SHORT_STAGE.max_steps,
        method=SHORT_STAGE.method,
    )
    long_stage = StageConfig(
        name=LONG_STAGE.name, sweep_name=LONG_STAGE.sweep_name, run_cap=resolve_run_cap(LONG_STAGE.run_cap, total_agents), max_steps=LONG_STAGE.max_steps, method=LONG_STAGE.method
    )

    if short_stage.run_cap != SHORT_STAGE.run_cap:
        print(f"Short stage run_cap increased from {SHORT_STAGE.run_cap} to {short_stage.run_cap} " f"to ensure at least {MIN_WAVES_PER_STAGE} waves across {total_agents} agents.")
    if long_stage.run_cap != LONG_STAGE.run_cap:
        print(f"Long stage run_cap increased from {LONG_STAGE.run_cap} to {long_stage.run_cap} " f"to ensure at least {MIN_WAVES_PER_STAGE} waves across {total_agents} agents.")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    install_signal_handlers()

    stages = [short_stage, long_stage]
    try:
        for stage in stages:
            run_stage(stage)
        write_overall_best_summary(stages)
        print("Two-stage article-style sweep finished.")
    finally:
        cleanup_processes()


if __name__ == "__main__":
    main()
