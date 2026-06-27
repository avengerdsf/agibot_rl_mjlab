"""Evaluate a trained policy while freezing selected swing-leg action channels."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

FreezeKind = Literal["none", "swing_hip_yaw", "swing_hip_roll", "swing_ankle_roll"]
FreezeMode = Literal["zero", "hold"]

FREEZE_PATTERNS: dict[FreezeKind, tuple[str, str] | None] = {
  "none": None,
  "swing_hip_yaw": ("left_hip_yaw_.*", "right_hip_yaw_.*"),
  "swing_hip_roll": ("left_hip_roll_.*", "right_hip_roll_.*"),
  "swing_ankle_roll": ("left_ankle_roll_.*", "right_ankle_roll_.*"),
}

SUMMARY_KEYS = (
  "Metrics/vel_diag/body/wz_abs_error_mean",
  "Metrics/vel_diag/hlip/pelvis_wz_abs_error_mean",
  "Metrics/vel_diag/hlip/swing_wz_abs_error_mean",
  "Metrics/hlip_swing_foot/yaw_omega_actual_abs_mean",
  "Metrics/hlip_swing_foot/yaw_omega_error_abs_mean",
  "Metrics/hlip_swing_foot/roll_omega_error_abs_mean",
  "Metrics/stance_contact_diag/full/swing_yaw_omega_abs_mean",
  "Metrics/stance_contact_diag/missing/swing_yaw_omega_abs_mean",
)


@dataclass(frozen=True)
class EvalFreezeConfig:
  checkpoint_file: str
  num_envs: int = 4096
  num_steps: int = 2000
  warmup_steps: int = 100
  device: str | None = None
  freeze: FreezeKind = "none"
  freeze_mode: FreezeMode = "zero"
  output_file: str | None = None
  no_terminations: bool = False


def _match_one(target_names: list[str], pattern: str) -> int:
  matches = [idx for idx, name in enumerate(target_names) if re.fullmatch(pattern, name)]
  if len(matches) != 1:
    raise RuntimeError(f"Expected one action target for {pattern}, got {matches}.")
  return matches[0]


def freeze_action_indices(
  target_names: list[str],
  freeze: FreezeKind,
  swing_idx: torch.Tensor,
  device: torch.device,
) -> torch.Tensor:
  patterns = FREEZE_PATTERNS[freeze]
  if patterns is None:
    return torch.full_like(swing_idx, -1, device=device)
  left_idx = _match_one(target_names, patterns[0])
  right_idx = _match_one(target_names, patterns[1])
  left = torch.full_like(swing_idx, left_idx, device=device)
  right = torch.full_like(swing_idx, right_idx, device=device)
  return torch.where(swing_idx == 0, left, right)


def apply_freeze_to_actions(
  actions: torch.Tensor,
  freeze: FreezeKind,
  target_names: list[str],
  swing_idx: torch.Tensor,
  mode: FreezeMode,
  previous_actions: torch.Tensor | None = None,
) -> torch.Tensor:
  if freeze == "none":
    return actions
  if mode == "hold" and previous_actions is None:
    raise RuntimeError("previous_actions is required when freeze_mode='hold'.")

  frozen = actions.clone()
  env_ids = torch.arange(actions.shape[0], device=actions.device)
  action_ids = freeze_action_indices(target_names, freeze, swing_idx, actions.device)
  if mode == "hold":
    assert previous_actions is not None
    frozen[env_ids, action_ids] = previous_actions[env_ids, action_ids]
  else:
    frozen[env_ids, action_ids] = 0.0
  return frozen


def _to_float(value) -> float:
  if isinstance(value, torch.Tensor):
    return float(value.detach().float().mean().cpu())
  return float(value)


def _collect_log_means(log: dict[str, object]) -> dict[str, float]:
  return {key: _to_float(value) for key, value in log.items() if key.startswith("Metrics/")}


def _print_summary(results: dict[str, dict[str, float]]) -> None:
  header = ["freeze", *[key.removeprefix("Metrics/") for key in SUMMARY_KEYS]]
  rows = []
  for freeze_name, values in results.items():
    rows.append([freeze_name, *[f"{values.get(key, 0.0):.4f}" for key in SUMMARY_KEYS]])
  widths = [
    max(len(str(row[idx])) for row in ([header] + rows))
    for idx in range(len(header))
  ]
  print("  ".join(text.ljust(widths[idx]) for idx, text in enumerate(header)))
  for row in rows:
    print("  ".join(text.ljust(widths[idx]) for idx, text in enumerate(row)))


def _make_policy(task_id: str, cfg: EvalFreezeConfig, env: RslRlVecEnvWrapper, device: str):
  agent_cfg = load_rl_cfg(task_id)
  checkpoint = Path(cfg.checkpoint_file).resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def run_one(task_id: str, cfg: EvalFreezeConfig, freeze: FreezeKind) -> dict[str, float]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  if cfg.no_terminations:
    env_cfg.terminations = {}

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  agent_cfg = load_rl_cfg(task_id)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = _make_policy(task_id, cfg, vec_env, device)
  action_term = vec_env.unwrapped.action_manager.get_term("joint_pos")
  target_names = list(action_term.target_names)
  command_term = vec_env.unwrapped.command_manager.get_term("hlip_ref")
  previous_actions = torch.zeros(
    (vec_env.num_envs, vec_env.num_actions),
    device=vec_env.device,
  )

  obs = vec_env.get_observations()
  metric_sums: dict[str, float] = {}
  metric_count = 0
  try:
    for step_idx in range(cfg.num_steps):
      with torch.no_grad():
        actions = policy(obs)
      actions = apply_freeze_to_actions(
        actions,
        freeze=freeze,
        target_names=target_names,
        swing_idx=command_term.swing_idx,
        mode=cfg.freeze_mode,
        previous_actions=previous_actions,
      )
      previous_actions = actions.detach().clone()
      obs, _, _, extras = vec_env.step(actions)
      if step_idx >= cfg.warmup_steps:
        log = extras.get("log", vec_env.unwrapped.extras.get("log", {}))
        for key, value in _collect_log_means(log).items():
          metric_sums[key] = metric_sums.get(key, 0.0) + value
        metric_count += 1
    if metric_count == 0:
      return {}
    return {key: value / metric_count for key, value in metric_sums.items()}
  finally:
    vec_env.close()


def run_eval(task_id: str, cfg: EvalFreezeConfig) -> dict[str, dict[str, float]]:
  results = {cfg.freeze: run_one(task_id, cfg, cfg.freeze)}
  _print_summary(results)
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] wrote {output_path}")
  return results


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import agibot_rl.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvalFreezeConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_eval(chosen_task, args)


if __name__ == "__main__":
  main()
