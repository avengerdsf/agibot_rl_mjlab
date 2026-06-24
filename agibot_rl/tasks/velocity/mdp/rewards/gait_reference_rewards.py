from __future__ import annotations

import re

import torch


def _matched_weights(
  names: list[str],
  patterns: dict[str, float] | None,
  device: torch.device,
) -> torch.Tensor:
  weights = torch.ones(len(names), device=device, dtype=torch.float32)
  if patterns is None:
    return weights
  for idx, name in enumerate(names):
    for pattern, value in patterns.items():
      if re.fullmatch(pattern, name):
        weights[idx] = value
        break
  return weights


def _matched_values(
  names: list[str],
  patterns: dict[str, float],
  default: float,
  device: torch.device,
) -> torch.Tensor:
  values = torch.full((len(names),), default, device=device, dtype=torch.float32)
  for idx, name in enumerate(names):
    for pattern, value in patterns.items():
      if re.fullmatch(pattern, name):
        values[idx] = value
        break
  return values


class x1_joint_default_pos:
  def __init__(self, cfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names)

    self.asset_name = asset_cfg.name
    self.joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    self.default_joint_pos = asset.data.default_joint_pos[:, self.joint_ids].clone()
    yaw_roll_patterns = cfg.params["yaw_roll_joint_names"]
    self.yaw_roll_ids = [
      torch.as_tensor(ids, device=env.device, dtype=torch.long)
      for ids, _ in (asset.find_joints(patterns) for patterns in yaw_roll_patterns)
    ]
    self.log_prefix = cfg.params.get("log_prefix", "Metrics/x1_joint_default_pos")

  def __call__(
    self,
    env,
    asset_cfg,
    yaw_roll_joint_names,
    yaw_roll_std: float = 0.5,
    joint_diff_scale: float = 0.01,
    log_prefix: str = "Metrics/x1_joint_default_pos",
  ) -> torch.Tensor:
    del asset_cfg, yaw_roll_joint_names, log_prefix

    asset = env.scene[self.asset_name]
    current_joint_pos = asset.data.joint_pos[:, self.joint_ids]
    error = current_joint_pos - self.default_joint_pos

    yaw_roll_error = torch.zeros(env.num_envs, device=env.device)
    for joint_ids in self.yaw_roll_ids:
      group_error = (
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
      )
      yaw_roll_error += torch.norm(group_error, dim=1)

    env.extras.setdefault("log", {})
    env.extras["log"][f"{self.log_prefix}_yaw_roll_error"] = torch.mean(
      yaw_roll_error
    )
    env.extras["log"][f"{self.log_prefix}_joint_error"] = torch.mean(
      torch.norm(error, dim=1)
    )

    gaussian_reward = torch.exp(
      -0.5 * torch.square(yaw_roll_error / yaw_roll_std)
    )
    return gaussian_reward - joint_diff_scale * torch.norm(error, dim=1)


class x1_joint_vel_l2:
  def __init__(self, cfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names)

    self.asset_name = asset_cfg.name
    self.joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    self.log_prefix = cfg.params.get("log_prefix", "Metrics/x1_joint_vel_l2")

  def __call__(
    self,
    env,
    asset_cfg,
    log_prefix: str = "Metrics/x1_joint_vel_l2",
  ) -> torch.Tensor:
    del asset_cfg, log_prefix

    asset = env.scene[self.asset_name]
    joint_vel = asset.data.joint_vel[:, self.joint_ids]
    penalty = torch.mean(torch.square(joint_vel), dim=1)

    env.extras.setdefault("log", {})
    env.extras["log"][f"{self.log_prefix}_abs_mean"] = torch.mean(torch.abs(joint_vel))
    return penalty


class swing_leg_yaw_roll_vel_l2:
  def __init__(self, cfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]

    self.asset_name = asset_cfg.name
    self.command_name = cfg.params["command_name"]
    self.log_prefix = cfg.params.get(
      "log_prefix",
      "Metrics/swing_leg_yaw_roll_vel_l2",
    )
    self.joint_weights = cfg.params.get(
      "joint_weights",
      {
        "hip_yaw": 1.0,
        "hip_roll": 1.0,
        "ankle_roll": 0.5,
      },
    )
    self.joint_groups = (
      ("hip_yaw", ("left_hip_yaw_.*",), ("right_hip_yaw_.*",)),
      ("hip_roll", ("left_hip_roll_.*",), ("right_hip_roll_.*",)),
      ("ankle_roll", ("left_ankle_roll_.*",), ("right_ankle_roll_.*",)),
    )
    self.left_joint_ids: dict[str, torch.Tensor] = {}
    self.right_joint_ids: dict[str, torch.Tensor] = {}
    for group_name, left_patterns, right_patterns in self.joint_groups:
      left_ids, _ = asset.find_joints(left_patterns)
      right_ids, _ = asset.find_joints(right_patterns)
      self.left_joint_ids[group_name] = torch.as_tensor(
        left_ids,
        device=env.device,
        dtype=torch.long,
      )
      self.right_joint_ids[group_name] = torch.as_tensor(
        right_ids,
        device=env.device,
        dtype=torch.long,
      )

  def __call__(
    self,
    env,
    asset_cfg,
    command_name: str,
    joint_weights: dict[str, float] | None = None,
    log_prefix: str = "Metrics/swing_leg_yaw_roll_vel_l2",
  ) -> torch.Tensor:
    del asset_cfg, command_name, joint_weights, log_prefix

    asset = env.scene[self.asset_name]
    command_term = env.command_manager.get_term(self.command_name)
    swing_idx = command_term.swing_idx
    joint_vel = asset.data.joint_vel
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=joint_vel.dtype)

    env.extras.setdefault("log", {})
    for group_name, _, _ in self.joint_groups:
      left_ids = self.left_joint_ids[group_name]
      right_ids = self.right_joint_ids[group_name]
      if left_ids.numel() == 0 or right_ids.numel() == 0:
        continue

      left_vel = joint_vel[:, left_ids]
      right_vel = joint_vel[:, right_ids]
      left_l2 = torch.mean(torch.square(left_vel), dim=1)
      right_l2 = torch.mean(torch.square(right_vel), dim=1)
      selected_l2 = torch.where(swing_idx == 0, left_l2, right_l2)
      penalty = penalty + float(self.joint_weights[group_name]) * selected_l2

      left_abs = torch.mean(torch.abs(left_vel), dim=1)
      right_abs = torch.mean(torch.abs(right_vel), dim=1)
      selected_abs = torch.where(swing_idx == 0, left_abs, right_abs)
      env.extras["log"][f"{self.log_prefix}/{group_name}_abs_mean"] = torch.mean(
        selected_abs
      )

    env.extras["log"][f"{self.log_prefix}/penalty_mean"] = torch.mean(penalty)
    return penalty


def gait_reference_joint_pos(
  env,
  command_name: str,
  std: float,
  joint_weights: dict[str, float] | None = None,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None, f"Command '{command_name}' not found."
  if not hasattr(command_term, "ref_joint_pos"):
    raise RuntimeError(
      "gait_reference_joint_pos requires a command term with ref_joint_pos."
    )

  asset = command_term.robot
  joint_ids = command_term.ref_joint_ids_tensor
  current_joint_pos = asset.data.joint_pos[:, joint_ids]
  ref_joint_pos = command_term.ref_joint_pos
  weights = _matched_weights(command_term.ref_joint_names, joint_weights, env.device)

  error = torch.square(current_joint_pos - ref_joint_pos)
  weighted_error = torch.sum(error * weights.unsqueeze(0), dim=1) / torch.clamp(
    weights.sum(), min=1.0
  )

  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/gait_reference_joint_error"] = torch.mean(
    torch.sqrt(weighted_error)
  )

  return torch.exp(-weighted_error / std**2)


def _log_scalar(log: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
  log[key] = value.detach() if isinstance(value, torch.Tensor) else value


def _log_mean(log: dict[str, torch.Tensor], key: str, value: torch.Tensor | None) -> None:
  if isinstance(value, torch.Tensor):
    log[key] = torch.mean(value.detach())


def _log_mean_abs(log: dict[str, torch.Tensor], key: str, value: torch.Tensor | None) -> None:
  if isinstance(value, torch.Tensor):
    log[key] = torch.mean(torch.abs(value.detach()))


def _matching_action_ids(target_names: tuple[str, ...] | list[str], patterns: tuple[str, ...]) -> list[int]:
  return [
    idx
    for idx, name in enumerate(target_names)
    if any(re.fullmatch(pattern, name) for pattern in patterns)
  ]


def _side_mean_abs(
  values: torch.Tensor,
  left_ids: list[int],
  right_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
  if not left_ids or not right_ids:
    return None
  left = torch.mean(torch.abs(values[:, left_ids]), dim=1)
  right = torch.mean(torch.abs(values[:, right_ids]), dim=1)
  return left, right


def _select_side_value(
  left: torch.Tensor,
  right: torch.Tensor,
  side_idx: torch.Tensor,
) -> torch.Tensor:
  return torch.where(side_idx == 0, left, right)


def _log_swing_yaw_source_diagnostics(env, command_term, log: dict[str, torch.Tensor]) -> None:
  dy_act = getattr(command_term, "dy_act", None)
  dy_out = getattr(command_term, "dy_out", None)
  swing_idx = getattr(command_term, "swing_idx", None)
  stance_idx = getattr(command_term, "stance_idx", None)
  robot = getattr(command_term, "robot", None)
  if (
    not isinstance(dy_act, torch.Tensor)
    or not isinstance(dy_out, torch.Tensor)
    or dy_act.dim() != 2
    or dy_out.shape != dy_act.shape
    or dy_act.shape[1] <= 11
    or not isinstance(swing_idx, torch.Tensor)
    or not isinstance(stance_idx, torch.Tensor)
    or robot is None
    or not hasattr(robot, "find_joints")
  ):
    return

  prefix = "Metrics/hlip_swing_yaw_source"
  log[f"{prefix}/swing_foot_yaw_rate_actual_abs_mean"] = torch.mean(torch.abs(dy_act[:, 11]))
  log[f"{prefix}/swing_foot_yaw_rate_ref_abs_mean"] = torch.mean(torch.abs(dy_out[:, 11]))
  log[f"{prefix}/swing_foot_yaw_rate_error_abs_mean"] = torch.mean(
    torch.abs(dy_act[:, 11] - dy_out[:, 11])
  )

  groups = (
    ("hip_yaw", ("left_hip_yaw_.*",), ("right_hip_yaw_.*",)),
    ("hip_roll", ("left_hip_roll_.*",), ("right_hip_roll_.*",)),
    ("ankle_roll", ("left_ankle_roll_.*",), ("right_ankle_roll_.*",)),
  )
  joint_vel = getattr(getattr(robot, "data", None), "joint_vel", None)
  if isinstance(joint_vel, torch.Tensor):
    for group_name, left_patterns, right_patterns in groups:
      left_ids, _ = robot.find_joints(left_patterns)
      right_ids, _ = robot.find_joints(right_patterns)
      side_values = _side_mean_abs(joint_vel, left_ids, right_ids)
      if side_values is None:
        continue
      left, right = side_values
      log[f"{prefix}/{group_name}/swing_joint_vel_abs_mean"] = torch.mean(
        _select_side_value(left, right, swing_idx)
      )
      log[f"{prefix}/{group_name}/stance_joint_vel_abs_mean"] = torch.mean(
        _select_side_value(left, right, stance_idx)
      )

  action_manager = getattr(env, "action_manager", None)
  if action_manager is None:
    return
  try:
    action_term = action_manager.get_term("joint_pos")
  except (KeyError, AttributeError):
    return
  raw_action = getattr(action_term, "raw_action", None)
  target_names = getattr(action_term, "target_names", None)
  if not isinstance(raw_action, torch.Tensor) or target_names is None:
    return
  for group_name, left_patterns, right_patterns in groups:
    left_ids = _matching_action_ids(target_names, left_patterns)
    right_ids = _matching_action_ids(target_names, right_patterns)
    side_values = _side_mean_abs(raw_action, left_ids, right_ids)
    if side_values is None:
      continue
    left, right = side_values
    log[f"{prefix}/{group_name}/swing_action_abs_mean"] = torch.mean(
      _select_side_value(left, right, swing_idx)
    )
    log[f"{prefix}/{group_name}/stance_action_abs_mean"] = torch.mean(
      _select_side_value(left, right, stance_idx)
    )


def _trace_value(
  value: torch.Tensor | None,
  env_id: int,
) -> torch.Tensor | None:
  if value is None or not isinstance(value, torch.Tensor):
    return None
  if value.dim() == 0:
    return value
  if value.shape[0] <= env_id:
    return None
  return value[env_id]


def _log_hlip_single_env_trace(env, command_term) -> None:
  y_act = getattr(command_term, "y_act", None)
  y_ref = getattr(command_term, "y_out", None)
  dy_act = getattr(command_term, "dy_act", None)
  dy_ref = getattr(command_term, "dy_out", None)
  if (
    not isinstance(y_act, torch.Tensor)
    or not isinstance(y_ref, torch.Tensor)
    or not isinstance(dy_act, torch.Tensor)
    or not isinstance(dy_ref, torch.Tensor)
    or y_act.dim() != 2
    or y_ref.shape != y_act.shape
    or dy_act.shape != y_act.shape
    or dy_ref.shape != y_act.shape
    or y_act.shape[1] < 12
  ):
    return

  trace_env_id = int(getattr(command_term, "hlip_trace_env_id", 0))
  if trace_env_id < 0 or trace_env_id >= y_act.shape[0]:
    return

  env.extras.setdefault("log", {})
  log = env.extras["log"]
  prefix = f"Metrics/hlip_trace/env{trace_env_id}"
  xyz_names = ("x", "y", "z")

  phase_fields = (
    "phase",
    "phase_var",
    "cur_swing_time",
    "stance_idx",
    "swing_idx",
  )
  for field in phase_fields:
    value = _trace_value(getattr(command_term, field, None), trace_env_id)
    if value is not None:
      _log_scalar(log, f"{prefix}/{field}", value)

  command = getattr(command_term, "last_command", None)
  command_value = _trace_value(command, trace_env_id)
  if command_value is not None and command_value.numel() >= 3:
    _log_scalar(log, f"{prefix}/command_x", command_value[0])
    _log_scalar(log, f"{prefix}/command_y", command_value[1])
    _log_scalar(log, f"{prefix}/command_yaw", command_value[2])
  if isinstance(command, torch.Tensor) and command.dim() == 2 and command.shape[1] >= 3:
    _log_mean(log, "Metrics/hlip_trace/mean/command_x", command[:, 0])
    _log_mean(log, "Metrics/hlip_trace/mean/command_y", command[:, 1])
    _log_mean(log, "Metrics/hlip_trace/mean/command_yaw", command[:, 2])
    _log_mean_abs(log, "Metrics/hlip_trace/mean_abs/command_yaw", command[:, 2])

  hlip_command = getattr(command_term, "last_hlip_command", None)
  hlip_command_value = _trace_value(hlip_command, trace_env_id)
  if hlip_command_value is not None and hlip_command_value.numel() >= 3:
    _log_scalar(log, f"{prefix}/hlip_command_x", hlip_command_value[0])
    _log_scalar(log, f"{prefix}/hlip_command_y", hlip_command_value[1])
    _log_scalar(log, f"{prefix}/hlip_command_yaw", hlip_command_value[2])
  if isinstance(hlip_command, torch.Tensor) and hlip_command.dim() == 2 and hlip_command.shape[1] >= 3:
    _log_mean(log, "Metrics/hlip_trace/mean/hlip_command_x", hlip_command[:, 0])
    _log_mean(log, "Metrics/hlip_trace/mean/hlip_command_y", hlip_command[:, 1])
    _log_mean(log, "Metrics/hlip_trace/mean/hlip_command_yaw", hlip_command[:, 2])
    _log_mean_abs(log, "Metrics/hlip_trace/mean_abs/hlip_command_yaw", hlip_command[:, 2])

  for idx, axis in enumerate(xyz_names):
    com_vel_error = dy_act[:, idx] - dy_ref[:, idx]
    _log_mean_abs(log, f"Metrics/hlip_trace/mean_abs/com_vel_error/{axis}", com_vel_error)

  metrics = getattr(command_term, "metrics", {})
  metric_names = (
    ("step_ref_x", "step_ref_x"),
    ("step_ref_y", "step_ref_y"),
    ("landing_actual_x", "landing/actual_x"),
    ("landing_actual_y", "landing/actual_y"),
    ("landing_target_x", "landing/target_x"),
    ("landing_target_y", "landing/target_y"),
    ("landing_error_x", "landing/error_x"),
    ("landing_error_y", "landing/error_y"),
    ("landing_valid", "landing/valid"),
  )
  for source_name, trace_name in metric_names:
    value = _trace_value(metrics.get(source_name), trace_env_id)
    if value is not None:
      _log_scalar(log, f"{prefix}/{trace_name}", value)


def clf_reward(
  env,
  command_name: str,
  max_eta_err: float,
  eps: float = 1e-6,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  max_clf = command_term.clf.lambda_max * max_eta_err**2 + eps
  reward = torch.exp(-torch.clamp(command_term.v, max=5.0 * max_clf) / max_clf)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_clf_v"] = torch.mean(command_term.v)
  _log_hlip_single_env_trace(env, command_term)
  y_err = getattr(
    command_term.clf,
    "last_y_err",
    torch.zeros(0, device=command_term.v.device),
  )
  dy_err = getattr(
    command_term.clf,
    "last_dy_err",
    torch.zeros(0, device=command_term.v.device),
  )
  if y_err.numel() > 0 and dy_err.numel() > 0:
    y_err_abs = torch.abs(y_err)
    dy_err_abs = torch.abs(dy_err)
    dy_act = getattr(
      command_term,
      "dy_act",
      torch.zeros(0, device=command_term.v.device),
    )
    dy_ref = getattr(
      command_term,
      "dy_out",
      torch.zeros(0, device=command_term.v.device),
    )
    env.extras["log"]["Metrics/hlip_clf/mean_abs_y_err"] = torch.mean(y_err_abs)
    env.extras["log"]["Metrics/hlip_clf/mean_abs_dy_err"] = torch.mean(dy_err_abs)
    if dy_act.shape == dy_err.shape and dy_ref.shape == dy_err.shape:
      dy_act_abs = torch.abs(dy_act)
      dy_ref_abs = torch.abs(dy_ref)
      env.extras["log"]["Metrics/hlip_clf/mean_abs_dy_act"] = torch.mean(dy_act_abs)
      env.extras["log"]["Metrics/hlip_clf/mean_abs_dy_ref"] = torch.mean(dy_ref_abs)
    env.extras["log"]["Metrics/hlip_clf/pelvis_yaw_err_abs_mean"] = torch.mean(
      y_err_abs[:, 5]
    )
    env.extras["log"]["Metrics/hlip_clf/swing_foot_yaw_err_abs_mean"] = torch.mean(
      y_err_abs[:, 11]
    )
    env.extras["log"]["Metrics/hlip_clf/pelvis_yaw_rate_err_abs_mean"] = torch.mean(
      dy_err_abs[:, 5]
    )
    env.extras["log"]["Metrics/hlip_clf/swing_foot_yaw_rate_err_abs_mean"] = torch.mean(
      dy_err_abs[:, 11]
    )
    _log_swing_yaw_source_diagnostics(env, command_term, env.extras["log"])
  return reward


def clf_decreasing_condition(
  env,
  command_name: str,
  alpha: float,
  eta_max: float,
  eta_dot_max: float,
  eps: float = 1e-6,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  max_violation = (
    2.0 * command_term.clf.norm_p * eta_max * eta_dot_max
    + alpha * command_term.clf.lambda_max * eta_max**2
    + eps
  )
  violation = torch.clamp(command_term.vdot + alpha * command_term.v, min=0.0)
  penalty = torch.clamp(violation / max_violation, min=0.0, max=1.0)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_clf_decay_violation"] = torch.mean(violation)
  env.extras["log"]["Metrics/hlip_clf/vdot_plus_alpha_v"] = torch.mean(
    command_term.vdot + alpha * command_term.v
  )
  return penalty


def hlip_upper_body_vel_error(
  env,
  command_name: str,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  ref_joint_vel = command_term.ref_joint_vel
  if ref_joint_vel.shape != command_term.upper_body_joint_vel.shape:
    ref_joint_vel = command_term.dy_out[:, 12:]
  vel_error = command_term.upper_body_joint_vel - ref_joint_vel
  penalty = torch.mean(torch.square(vel_error), dim=1)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_upper_body_vel_error"] = torch.mean(
    torch.sqrt(penalty)
  )
  return penalty


def hlip_yaw_rate_error(
  env,
  command_name: str,
  root_weight: float,
  pelvis_weight: float,
  swing_weight: float,
  max_penalty: float,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  root_error = (
    command_term.robot.data.root_link_ang_vel_b[:, 2]
    - command_term.last_command[:, 2]
  )
  yaw_rate_error = command_term.dy_act[:, [5, 11]] - command_term.dy_out[:, [5, 11]]
  pelvis_error = yaw_rate_error[:, 0]
  swing_error = yaw_rate_error[:, 1]
  penalty = (
    root_weight * torch.square(root_error)
    + pelvis_weight * torch.square(pelvis_error)
    + swing_weight * torch.square(swing_error)
  )
  penalty = torch.clamp(penalty, max=max_penalty)

  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_yaw_rate_error/root_abs_mean"] = torch.mean(
    torch.abs(root_error)
  )
  env.extras["log"]["Metrics/hlip_yaw_rate_error/pelvis_abs_mean"] = torch.mean(
    torch.abs(pelvis_error)
  )
  env.extras["log"]["Metrics/hlip_yaw_rate_error/swing_abs_mean"] = torch.mean(
    torch.abs(swing_error)
  )
  env.extras["log"]["Metrics/hlip_yaw_rate_error/penalty_mean"] = torch.mean(penalty)
  return penalty


def holonomic_constraint(
  env,
  command_name: str,
  sigma_pose: float,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  delta_xy = command_term.stance_foot_pos[:, :2] - command_term.stance_foot_pos_0[:, :2]
  delta_z = (
    command_term.stance_foot_pos[:, 2] - command_term.stance_foot_pos_0[:, 2]
  ).unsqueeze(1)
  delta_roll = (
    command_term.stance_foot_ori[:, 0]
    - command_term.stance_foot_ori_0[:, 0]
    + torch.pi
  ) % (2.0 * torch.pi) - torch.pi
  delta_yaw = (
    command_term.stance_foot_ori[:, 2] - command_term.stance_foot_ori_0[:, 2] + torch.pi
  ) % (2.0 * torch.pi) - torch.pi
  pose_error = torch.cat(
    (delta_xy, delta_z, delta_roll.unsqueeze(1), delta_yaw.unsqueeze(1)), dim=1
  )
  error_norm = torch.sum(torch.square(pose_error), dim=1)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_holonomic/pose_error"] = torch.mean(
    torch.sqrt(error_norm)
  )
  return command_term.get_not_flight_envs() * torch.exp(-error_norm / sigma_pose**2)


def holonomic_constraint_vel(
  env,
  command_name: str,
  sigma_vel: float,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  yaw_rate = command_term.stance_foot_rpy_rate[:, 2].unsqueeze(1)
  vel_error = torch.cat((command_term.stance_foot_vel, yaw_rate), dim=1)
  error_norm = torch.sum(torch.square(vel_error), dim=1)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/hlip_holonomic/vel_error"] = torch.mean(
    torch.sqrt(error_norm)
  )
  return command_term.get_not_flight_envs() * torch.exp(-error_norm / sigma_vel**2)
