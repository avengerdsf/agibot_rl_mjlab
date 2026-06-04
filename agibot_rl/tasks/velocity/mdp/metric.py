from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_orientation_l2_raw(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """计算和 body_orientation_l2 reward 一致的姿态误差。"""

  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    body_quat_w = body_quat_w.squeeze(1)

    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_tilt_angle_deg(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """返回 base 倾角，单位 degree。

  该值和 body_orientation_l2 使用同一套姿态计算逻辑。
  """

  orientation_l2 = _body_orientation_l2_raw(env, asset_cfg=asset_cfg)
  orientation_l2 = torch.clamp(orientation_l2, 0.0, 1.0)

  tilt_angle = torch.asin(torch.sqrt(orientation_l2))
  return torch.rad2deg(tilt_angle)


def bad_orientation_flag(
  env: "ManagerBasedRlEnv",
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """返回姿态是否超过阈值。

  返回 1 表示超过阈值。
  返回 0 表示没有超过阈值。
  """

  orientation_l2 = _body_orientation_l2_raw(env, asset_cfg=asset_cfg)
  threshold = math.sin(limit_angle) ** 2

  return (orientation_l2 > threshold).float()


def body_height(
  env: "ManagerBasedRlEnv",
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    height =  asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    height =  asset.data.root_link_pos_w[:, 2]
  error = torch.square(height - target_height)

  return error


def actuator_force_ratio(
  env,
  sensor_names: list[str],
  effort_limits: list[float],
  threshold: float = 0.9,
  log_prefix: str = "Metrics/actuator_force_ratio",
  log_per_sensor: bool = False,
):
  forces = []
  for sensor_name in sensor_names:
    sensor = env.scene[sensor_name]
    forces.append(sensor.data.reshape(env.num_envs, -1).squeeze(-1))

  force = torch.stack(forces, dim=1)
  limits = torch.tensor(effort_limits, device=env.device).view(1, -1)
  ratio = torch.abs(force) / limits

  env.extras.setdefault("log", {})
  env.extras["log"][f"{log_prefix}_max"] = ratio.max()
  env.extras["log"][f"{log_prefix}_mean"] = ratio.mean()
  env.extras["log"][f"{log_prefix}_saturation_fraction"] = (
    ratio > threshold
  ).float().mean()
  if log_per_sensor:
    for sensor_name, sensor_ratio in zip(sensor_names, ratio.unbind(dim=1)):
      metric_name = sensor_name.removeprefix("robot/jointeffort_")
      env.extras["log"][f"{log_prefix}/{metric_name}_mean"] = sensor_ratio.mean()
      env.extras["log"][f"{log_prefix}/{metric_name}_max"] = sensor_ratio.max()

  return ratio.max(dim=1).values


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  mask_f = mask.float()
  return torch.sum(value * mask_f) / torch.clamp(mask_f.sum(), min=1.0)


def _action_indices(target_names: list[str], patterns: list[str]) -> list[int]:
  indices = []
  for idx, name in enumerate(target_names):
    if any(re.fullmatch(pattern, name) for pattern in patterns):
      indices.append(idx)
  return indices


def _log_joint_group(
  env,
  asset: Entity,
  action_term,
  group_name: str,
  joint_patterns: list[str],
) -> None:
  joint_ids, _ = asset.find_joints(joint_patterns)
  if len(joint_ids) > 0:
    joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    env.extras["log"][f"Metrics/yaw/{group_name}_pos_abs_mean"] = torch.mean(
      torch.abs(asset.data.joint_pos[:, joint_ids_tensor])
    )
    env.extras["log"][f"Metrics/yaw/{group_name}_vel_abs_mean"] = torch.mean(
      torch.abs(asset.data.joint_vel[:, joint_ids_tensor])
    )

  action_ids = _action_indices(action_term.target_names, joint_patterns)
  if action_ids:
    action_ids_tensor = torch.as_tensor(action_ids, device=env.device, dtype=torch.long)
    env.extras["log"][f"Metrics/yaw/{group_name}_action_abs_mean"] = torch.mean(
      torch.abs(action_term.raw_action[:, action_ids_tensor])
    )


def yaw_tracking_diagnostics(
  env: "ManagerBasedRlEnv",
  command_name: str,
  action_name: str,
  vx_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  actual_wz = asset.data.root_link_ang_vel_b[:, 2]
  cmd_wz = command[:, 2]
  signed_error = cmd_wz - actual_wz
  abs_error = torch.abs(signed_error)
  command_speed_xy = torch.linalg.norm(command[:, :2], dim=1)
  low_vx = command_speed_xy <= vx_threshold
  high_vx = command_speed_xy > vx_threshold

  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/yaw/cmd_wz_abs_mean"] = torch.mean(torch.abs(cmd_wz))
  env.extras["log"]["Metrics/yaw/actual_wz_abs_mean"] = torch.mean(torch.abs(actual_wz))
  env.extras["log"]["Metrics/yaw/signed_error_wz_mean"] = torch.mean(signed_error)
  env.extras["log"]["Metrics/yaw/abs_error_wz_mean"] = torch.mean(abs_error)
  env.extras["log"]["Metrics/yaw/abs_error_wz_when_vx_low"] = _masked_mean(abs_error, low_vx)
  env.extras["log"]["Metrics/yaw/abs_error_wz_when_vx_high"] = _masked_mean(abs_error, high_vx)
  env.extras["log"]["Metrics/yaw/vx_low_fraction"] = low_vx.float().mean()
  env.extras["log"]["Metrics/yaw/vx_high_fraction"] = high_vx.float().mean()

  action_term = env.action_manager.get_term(action_name)
  _log_joint_group(env, asset, action_term, "lumbar_yaw", [r"lumbar_yaw_.*"])
  _log_joint_group(env, asset, action_term, "hip_yaw", [r".*_hip_yaw_.*"])
  _log_joint_group(env, asset, action_term, "ankle_roll", [r".*_ankle_roll_.*"])

  return abs_error
