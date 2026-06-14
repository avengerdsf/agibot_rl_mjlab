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

  hlip_command = getattr(command_term, "last_hlip_command", None)
  hlip_command_value = _trace_value(hlip_command, trace_env_id)
  if hlip_command_value is not None and hlip_command_value.numel() >= 3:
    _log_scalar(log, f"{prefix}/hlip_command_x", hlip_command_value[0])
    _log_scalar(log, f"{prefix}/hlip_command_y", hlip_command_value[1])
    _log_scalar(log, f"{prefix}/hlip_command_yaw", hlip_command_value[2])

  robot = getattr(command_term, "robot", None)
  robot_data = getattr(robot, "data", None)
  root_link_lin_vel_b = _trace_value(
    getattr(robot_data, "root_link_lin_vel_b", None),
    trace_env_id,
  )
  if root_link_lin_vel_b is not None and root_link_lin_vel_b.numel() >= 2:
    _log_scalar(log, f"{prefix}/root_link_vel_b/x", root_link_lin_vel_b[0])
    _log_scalar(log, f"{prefix}/root_link_vel_b/y", root_link_lin_vel_b[1])
    if command_value is not None and command_value.numel() >= 2:
      _log_scalar(log, f"{prefix}/twist_error/x", root_link_lin_vel_b[0] - command_value[0])
      _log_scalar(log, f"{prefix}/twist_error/y", root_link_lin_vel_b[1] - command_value[1])

  root_com_vel_w = _trace_value(
    getattr(robot_data, "root_com_vel_w", None),
    trace_env_id,
  )
  if root_com_vel_w is not None and root_com_vel_w.numel() >= 3:
    _log_scalar(log, f"{prefix}/root_com_vel_w/x", root_com_vel_w[0])
    _log_scalar(log, f"{prefix}/root_com_vel_w/y", root_com_vel_w[1])

  for idx, axis in enumerate(xyz_names):
    _log_scalar(log, f"{prefix}/com_act/vel_{axis}", dy_act[trace_env_id, idx])
    _log_scalar(log, f"{prefix}/com_ref/vel_{axis}", dy_ref[trace_env_id, idx])
    _log_scalar(log, f"{prefix}/com_vel_error/{axis}", dy_act[trace_env_id, idx] - dy_ref[trace_env_id, idx])

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
