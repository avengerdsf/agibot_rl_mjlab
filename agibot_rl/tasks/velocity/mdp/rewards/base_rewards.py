from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class base_acc:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]

    self.asset_name = asset_cfg.name
    self.prev_root_lin_vel_b = asset.data.root_link_lin_vel_b.clone()
    self.log_prefix = cfg.params.get("log_prefix", "Metrics/base_acc")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    scale: float = 3.0,
    log_prefix: str = "Metrics/base_acc",
  ) -> torch.Tensor:
    del asset_cfg, log_prefix

    asset: Entity = env.scene[self.asset_name]
    root_lin_vel_b = asset.data.root_link_lin_vel_b
    root_acc = self.prev_root_lin_vel_b - root_lin_vel_b
    reset = env.episode_length_buf == 0
    if torch.any(reset):
      root_acc[reset] = 0.0

    root_acc_norm = torch.linalg.norm(root_acc, dim=1)
    self.prev_root_lin_vel_b = root_lin_vel_b.clone()

    env.extras.setdefault("log", {})
    env.extras["log"][f"{self.log_prefix}_mean"] = torch.mean(root_acc_norm)
    return torch.exp(-scale * root_acc_norm)


class base_acc_l2:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]

    self.asset_name = asset_cfg.name
    self.prev_root_lin_vel_b = asset.data.root_link_lin_vel_b.clone()
    self.log_prefix = cfg.params.get("log_prefix", "Metrics/base_acc")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    log_prefix: str = "Metrics/base_acc",
  ) -> torch.Tensor:
    del asset_cfg, log_prefix

    asset: Entity = env.scene[self.asset_name]
    root_lin_vel_b = asset.data.root_link_lin_vel_b
    root_acc = root_lin_vel_b - self.prev_root_lin_vel_b
    reset = env.episode_length_buf == 0
    if torch.any(reset):
      root_acc[reset] = 0.0

    root_acc_norm = torch.linalg.norm(root_acc, dim=1)
    root_acc_l2 = torch.sum(torch.square(root_acc), dim=1)
    self.prev_root_lin_vel_b = root_lin_vel_b.clone()

    env.extras.setdefault("log", {})
    env.extras["log"][f"{self.log_prefix}_mean"] = torch.mean(root_acc_norm)
    env.extras["log"][f"{self.log_prefix}_l2_mean"] = torch.mean(root_acc_l2)
    return root_acc_l2


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def track_vel_hard(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_v: float = 0.40,
  sigma_omega: float = 0.60,
  penalty_scale: float = 0.30,
  v_max: float = 1.0,
  omega_max: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  lin_vel = asset.data.root_link_lin_vel_b
  ang_vel = asset.data.root_link_ang_vel_b

  e_v = torch.linalg.norm(command[:, :2] - lin_vel[:, :2], dim=1)
  e_omega = torch.abs(command[:, 2] - ang_vel[:, 2])
  e_v_norm = e_v / v_max
  e_omega_norm = e_omega / omega_max

  return (
    0.5
    * (
      torch.exp(-torch.square(e_v / sigma_v))
      + torch.exp(-torch.square(e_omega / sigma_omega))
    )
    - penalty_scale * (e_v_norm + e_omega_norm)
  )


class base_heading_tracking:
  """Reward base heading tracking while avoiding unwanted yaw drift.

  Heading-command envs track the command term's explicit heading target. Other
  envs latch the current heading whenever a near-zero yaw command begins, then
  reward staying close to that latched heading.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.heading_target = torch.zeros(env.num_envs, device=env.device)
    self.prev_hold_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.5,
    command_threshold: float = 0.05,
    reward_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    command_term = env.command_manager.get_term(command_name)
    current_heading = asset.data.heading_w
    zero_yaw_command = torch.abs(command[:, 2]) <= command_threshold

    is_heading_env = getattr(command_term, "is_heading_env", None)
    if is_heading_env is None:
      is_heading_env = torch.zeros_like(zero_yaw_command)
    else:
      is_heading_env = is_heading_env.bool()

    is_standing_env = getattr(command_term, "is_standing_env", None)
    if is_standing_env is None:
      is_standing_env = torch.zeros_like(zero_yaw_command)
    else:
      is_standing_env = is_standing_env.bool()

    heading_command_active = is_heading_env & ~is_standing_env
    hold_active = zero_yaw_command & ~heading_command_active

    reset_like = env.episode_length_buf <= 1
    capture_target = hold_active & ((~self.prev_hold_active) | reset_like)
    self.heading_target = torch.where(
      capture_target,
      current_heading.detach(),
      self.heading_target,
    )

    explicit_heading_target = getattr(command_term, "heading_target", None)
    if explicit_heading_target is None:
      explicit_heading_target = self.heading_target
    target_heading = torch.where(
      heading_command_active,
      explicit_heading_target,
      self.heading_target,
    )

    active = hold_active | heading_command_active
    heading_error = wrap_to_pi(target_heading - current_heading)
    reward = reward_scale * torch.exp(-torch.abs(heading_error) / sigma)
    reward = reward * active.float()

    env.extras.setdefault("log", {})
    active_count = torch.clamp(active.float().sum(), min=1.0)
    env.extras["log"]["Metrics/base_heading_error_mean"] = (
      torch.sum(torch.abs(heading_error) * active.float()) / active_count
    )

    self.prev_hold_active = hold_active
    return reward


def yaw_rate_zero_command_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize yaw rate when the command does not ask for yaw rotation."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  active = (torch.abs(command[:, 2]) <= command_threshold).float()
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
  cost = torch.square(yaw_rate) * active

  env.extras.setdefault("log", {})
  active_count = torch.clamp(active.sum(), min=1.0)
  env.extras["log"]["Metrics/yaw_rate_zero_command_mean"] = (
    torch.sum(torch.abs(yaw_rate) * active) / active_count
  )
  return cost


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """奖励机器人保持身体竖直。

  如果 asset_cfg 指定了 body_ids，则计算指定 link 的姿态倾斜程度。
  如果没有指定 body_ids，则使用机器人 root link 的 projected_gravity_b。
  """
  asset: Entity = env.scene[asset_cfg.name]

  # 如果指定了 body_ids，就用指定 link 的四元数来计算该 link 坐标系下的重力方向
  if asset_cfg.body_ids:
    # 取出指定 body link 在世界系下的姿态四元数
    # 形状为 [B, N, 4]
    # B 表示并行环境数量，N 表示选中的 body 数量，4 表示四元数
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]

    # 当前这里只支持或默认使用一个 body_id
    # 将形状从 [B, 1, 4] 压缩成 [B, 4]
    body_quat_w = body_quat_w.squeeze(1)

    # 世界系下的重力方向向量
    # 通常表示为 [0, 0, -1] 或类似归一化方向
    gravity_w = asset.data.gravity_vec_w

    # 将世界系下的重力方向旋转到该 body link 的局部坐标系下
    # 如果 body 完全竖直，则局部系下的重力主要落在 z 轴
    # 如果 body 前后或左右倾斜，则重力会在局部 x、y 方向产生分量
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

    # 只取局部 x、y 方向的重力分量，并计算平方和
    # x、y 分量越大，说明身体越倾斜
    # dim=1 表示对每个环境内部的 x、y 两个分量求和
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

  else:
    # 如果没有指定 body_ids，就直接使用 root link 的局部重力投影
    # projected_gravity_b[:, :2] 表示 root link 局部 x、y 方向的重力分量
    # 平方和越大，说明 root link 越偏离竖直状态
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

  # 返回每个环境的身体倾斜惩罚值
  # 形状为 [B]
  # 值越接近 0，表示身体越竖直
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def angular_momentum_proxy_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize whole-body angular momentum without requiring a MuJoCo sensor.

    Computes the same quantity as a root-body ``subtreeangmom`` sensor:
    body rotational angular momentum plus mass-weighted translational angular
    momentum about the selected bodies' CoM. ``normalize`` is kept for
    backward-compatible reward configs and intentionally does not alter the
    value, so the reward scale matches ``angular_momentum_penalty``.
    """
    del normalize
    asset: Entity = env.scene[asset_cfg.name]

    body_sel = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    body_ids = asset.data.indexing.body_ids[body_sel]
    if body_ids.dim() == 0:
        body_ids = body_ids.unsqueeze(0)

    data = asset.data.data
    model = asset.data.model
    num_envs = data.xipos.shape[0]

    body_pos_w = data.xipos[:, body_ids]  # [B, N, 3]
    body_ximat_w = data.ximat[:, body_ids]  # [B, N, 3, 3]
    body_cvel = data.cvel[:, body_ids]  # [B, N, 6], rot:lin
    body_ang_vel_w = body_cvel[..., :3]

    body_root_ids = model.body_rootid[body_ids]
    body_root_com_w = data.subtree_com[:, body_root_ids]
    body_lin_vel_w = body_cvel[..., 3:6] - torch.linalg.cross(
        body_pos_w - body_root_com_w,
        body_ang_vel_w,
        dim=-1,
    )

    body_mass = model.body_mass
    if body_mass.dim() == 1:
        body_mass = body_mass[body_ids].unsqueeze(0)
    elif body_mass.dim() == 2:
        body_mass = body_mass[:, body_ids]
    else:
        raise RuntimeError(f"Unexpected body_mass shape: {body_mass.shape}")
    if body_mass.shape[0] != num_envs:
        env_model_ids = torch.arange(num_envs, device=body_mass.device) % body_mass.shape[0]
        body_mass = body_mass[env_model_ids]

    body_inertia = model.body_inertia
    if body_inertia.dim() == 2:
        body_inertia = body_inertia[body_ids].unsqueeze(0)
    elif body_inertia.dim() == 3:
        body_inertia = body_inertia[:, body_ids]
    else:
        raise RuntimeError(f"Unexpected body_inertia shape: {body_inertia.shape}")
    if body_inertia.shape[0] != num_envs:
        env_model_ids = torch.arange(num_envs, device=body_inertia.device) % body_inertia.shape[0]
        body_inertia = body_inertia[env_model_ids]

    mass = body_mass.unsqueeze(-1)
    total_mass = torch.clamp(mass.sum(dim=1, keepdim=True), min=eps)
    com_pos_w = (mass * body_pos_w).sum(dim=1, keepdim=True) / total_mass
    com_vel_w = (mass * body_lin_vel_w).sum(dim=1, keepdim=True) / total_mass

    rel_pos_w = body_pos_w - com_pos_w
    rel_lin_vel_w = body_lin_vel_w - com_vel_w
    translational_angmom = torch.linalg.cross(
        rel_pos_w,
        mass * rel_lin_vel_w,
        dim=-1,
    )

    body_ang_vel_inertia_frame = torch.matmul(
        body_ximat_w.transpose(-1, -2),
        body_ang_vel_w.unsqueeze(-1),
    ).squeeze(-1)
    rotational_angmom = torch.matmul(
        body_ximat_w,
        (body_inertia * body_ang_vel_inertia_frame).unsqueeze(-1),
    ).squeeze(-1)

    angmom = torch.sum(rotational_angmom + translational_angmom, dim=1)
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
    env.extras.setdefault("log", {})
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def swing_foot_trajectory(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  command_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  if not hasattr(command_term, "ref_swing_foot_pos_b"):
    raise RuntimeError(
      "swing_foot_trajectory requires a command term with ref_swing_foot_pos_b."
    )

  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
  root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(
    foot_pos_w.shape[0], foot_pos_w.shape[1], 4
  )
  foot_pos_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    (foot_pos_w - root_pos_w).reshape(-1, 3),
  ).reshape(foot_pos_w.shape)
  error = foot_pos_b - command_term.ref_swing_foot_pos_b
  swing_mask = command_term.swing_foot_mask.float()
  pos_error = torch.sum(torch.square(error), dim=-1)
  reward = torch.exp(-torch.sum(pos_error * swing_mask, dim=1) / std**2)
  command_cfg = getattr(command_term, "cfg", None)
  velocity_command_name = getattr(command_cfg, "velocity_command_name", command_name)
  command = env.command_manager.get_command(velocity_command_name)
  active = (torch.linalg.norm(command, dim=1) > command_threshold).float()
  reward = reward * active
  active_count = torch.clamp(torch.sum(swing_mask), min=1.0)
  env.extras["log"]["Metrics/swing_foot_trajectory_error"] = torch.sum(
    torch.sqrt(pos_error) * swing_mask
  ) / active_count
  for idx, name in enumerate(command_term.foot_body_names):
    env.extras["log"][f"Metrics/swing_foot_trajectory_error/{name}"] = torch.sum(
      torch.sqrt(pos_error[:, idx]) * swing_mask[:, idx]
    ) / torch.clamp(torch.sum(swing_mask[:, idx]), min=1.0)
  return reward


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def feet_contact_number(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.1,
  num_feet: int = 2,
) -> torch.Tensor:
  # 获取接触传感器
  sensor: ContactSensor = env.scene[sensor_name]

  # 获取速度指令，通常包含 vx、vy、yaw_rate
  command = env.command_manager.get_command(command_name)
  assert command is not None

  # 判断每个接触检测点当前是否处于接触状态
  # current_contact_time > 0.0 表示该点已经发生接触
  contact = sensor.data.current_contact_time > 0.0

  # 如果接触点数量不等于脚的数量，说明每只脚可能包含多个接触几何体
  if contact.shape[1] != num_feet:
    # 接触点总数必须能被脚数量整除，否则无法均分到每只脚
    if contact.shape[1] % num_feet != 0:
      raise RuntimeError(
        f"feet_contact_number expected contact count divisible by num_feet={num_feet}, "
        f"got {contact.shape[1]}."
      )

    # 计算每只脚对应的接触几何体数量
    geoms_per_foot = contact.shape[1] // num_feet

    # 将接触状态整理为 [num_envs, num_feet, geoms_per_foot]
    # 只要同一只脚下任意一个几何体接触，就认为该脚接触地面
    contact = contact.reshape(contact.shape[0], num_feet, geoms_per_foot).any(dim=-1)

  # 计算 xy 平面线速度指令模长
  linear_norm = torch.norm(command[:, :2], dim=1)

  # 计算 yaw 角速度指令绝对值
  angular_norm = torch.abs(command[:, 2])

  # 判断当前是否为站立状态
  # 当线速度和角速度都足够小时，认为机器人应该双脚接触地面
  standing = (linear_norm + angular_norm) <= command_threshold

  # 根据当前 episode 时间计算全局步态相位
  # global_phase 的范围会随时间递增，后面通过 % 1.0 转成周期相位
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)

  # 将每只脚的相位偏移转换为 tensor
  # offset 用于让不同脚处于不同步态相位
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)

  # 计算每只脚在当前相位下是否应该处于支撑状态
  # 小于 threshold 表示该脚处于期望接触区间
  stance_mask = ((global_phase + offsets) % 1.0) < threshold

  # 生成目标接触状态
  # 站立时目标为所有脚都接触地面
  # 行走时目标为按照步态相位接触
  target_contact = torch.where(standing.unsqueeze(1), torch.ones_like(stance_mask), stance_mask)

  # 对每只脚分别计算奖励
  # 实际接触状态等于目标接触状态时给 1
  # 不一致时给 -0.3
  per_foot_reward = torch.where(
    contact == target_contact,
    torch.ones_like(target_contact, dtype=global_phase.dtype),
    torch.full_like(target_contact, -0.3, dtype=global_phase.dtype),
  )

  # 对所有脚的奖励取平均，得到每个环境的最终奖励
  return per_foot_reward.mean(dim=1)

def body_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    height = asset.data.root_link_pos_w[:, 2]

  error = torch.square(height - target_height)
  reward = 1 - torch.exp(-error / std**2)

  return reward


class feet_swing_height:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.target_height = cfg.params["target_height"]
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    in_air_float = in_air.float()
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    landing_cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
    low_height_error = torch.clamp((target_height - foot_heights) / target_height, min=0.0)
    swing_cost = torch.sum(torch.square(low_height_error) * in_air_float, dim=1)
    cost = (landing_cost + swing_cost) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    env.extras["log"]["Metrics/swing_low_height_error_mean"] = torch.mean(
      low_height_error * in_air_float
    )
    for idx, name in enumerate(self.site_names):
      foot_first_contact = first_contact[:, idx].float()
      foot_num_landings = torch.sum(foot_first_contact)
      foot_peak_heights = self.peak_heights[:, idx] * foot_first_contact
      env.extras["log"][f"Metrics/peak_height/{name}"] = torch.sum(
        foot_peak_heights
      ) / torch.clamp(foot_num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_geom_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  num_feet: int = 2,
  period: float | None = None,
  offset: list[float] | None = None,
  threshold: float | None = None,
  min_contact_fraction: float = 0.0,
) -> torch.Tensor:
  """Penalize slip using foot collision geom velocities while those geoms contact terrain."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  phase_cfg_set = period is not None or offset is not None or threshold is not None
  if phase_cfg_set and (period is None or offset is None or threshold is None):
    raise RuntimeError(
      "feet_geom_slip requires period, offset, and threshold to be set together."
    )

  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()

  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  geom_vel_xy = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
  vel_xy_norm = torch.norm(geom_vel_xy, dim=-1)

  if in_contact.shape != vel_xy_norm.shape:
    raise RuntimeError(
      "feet_geom_slip requires contact sensor primaries and asset_cfg geoms to "
      f"match, got contact {in_contact.shape} and geoms {vel_xy_norm.shape}."
    )
  if in_contact.shape[1] % num_feet != 0:
    raise RuntimeError(
      f"feet_geom_slip expected geom count divisible by num_feet={num_feet}, "
      f"got {in_contact.shape[1]} geoms."
    )

  geoms_per_foot = in_contact.shape[1] // num_feet
  in_contact_by_foot = in_contact.reshape(in_contact.shape[0], num_feet, geoms_per_foot)
  vel_sq_by_foot = torch.square(vel_xy_norm).reshape(
    vel_xy_norm.shape[0], num_feet, geoms_per_foot
  )
  vel_by_foot = vel_xy_norm.reshape(vel_xy_norm.shape[0], num_feet, geoms_per_foot)

  contact_count = torch.sum(in_contact_by_foot, dim=-1)
  foot_has_contact = contact_count > 0
  contact_fraction = contact_count / geoms_per_foot
  foot_cost = torch.sum(vel_sq_by_foot * in_contact_by_foot, dim=-1) / torch.clamp(
    contact_count, min=1
  )

  foot_weight = torch.ones_like(foot_cost)
  if phase_cfg_set:
    assert period is not None and offset is not None and threshold is not None
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
      1, -1
    )
    foot_weight = (((global_phase + offsets) % 1.0) < threshold).float()

  stable_contact = contact_fraction >= min_contact_fraction
  active_foot = foot_has_contact & stable_contact
  cost = torch.sum(foot_cost * active_foot.float() * foot_weight, dim=1) * active

  metric_contact = in_contact_by_foot * foot_weight.unsqueeze(-1) * active_foot.unsqueeze(-1)
  num_in_contact = torch.sum(metric_contact)
  mean_slip_vel = torch.sum(vel_by_foot * metric_contact) / torch.clamp(
    num_in_contact, min=1.0
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  env.extras["log"]["Metrics/foot_contact_geom_count_mean"] = torch.mean(contact_count)
  return cost


def stance_foot_contact_count_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period: float,
  offset: list[float],
  threshold: float,
  command_name: str,
  command_threshold: float,
  num_feet: int = 2,
) -> torch.Tensor:
  """惩罚支撑脚足底接触不足，减少只用脚尖或少量 geom 支撑的行为。"""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert contact_sensor.data.found is not None

  in_contact = (contact_sensor.data.found > 0).float()
  if in_contact.shape[1] % num_feet != 0:
    raise RuntimeError(
      f"stance_foot_contact_count_penalty expected geom count divisible by num_feet={num_feet}, "
      f"got {in_contact.shape[1]} geoms."
    )

  # 接触传感器按足底 geom 记录接触状态，这里按左右脚重新分组并计算每只脚的接触比例。
  geoms_per_foot = in_contact.shape[1] // num_feet
  contact_count = in_contact.reshape(
    in_contact.shape[0], num_feet, geoms_per_foot
  ).sum(dim=-1)
  contact_fraction = contact_count / geoms_per_foot

  # 运动命令下按步态相位要求支撑脚接触，站立命令下要求双脚都接触地面。
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  moving = (linear_norm + angular_norm) > command_threshold

  # 每只脚的支撑相位由全局相位加各自 offset 得到。
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  stance_mask = ((global_phase + offsets) % 1.0) < threshold
  target_contact = torch.where(
    moving.unsqueeze(1),
    stance_mask,
    torch.ones_like(stance_mask),
  ).float()

  # 只惩罚应处于支撑期但接触不足的脚。
  missing_contact = (1.0 - contact_fraction) * target_contact
  target_count = torch.clamp(target_contact.sum(dim=1), min=1.0)
  cost = missing_contact.sum(dim=1) / target_count

  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/stance_foot_contact_fraction_mean"] = torch.sum(
    contact_fraction * target_contact
  ) / torch.clamp(target_contact.sum(), min=1.0)
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    # 获取关节初始位置信息
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    # 把 standing / walking / running 三套容忍度，按真实 joint 顺序展开成 tensor
    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    # 判断运动状态（站立/行走/跑步），根据命令速度划分到不同的容忍度等级
    command = env.command_manager.get_command(command_name)
    assert command is not None

    # command = [
    #     [vx, vy, wz],
    #     [vx, vy, wz],
    #     [vx, vy, wz],
    # ]
    linear_speed = torch.norm(command[:, :2], dim=1) # 计算每一行前两个元素的欧几里得范数，得到线速度的大小，结果是一个一维张量，每个元素对应一行的线速度大小。
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    # 计算当前关节角和默认关节角的误差
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    # 先用 std^2 归一化误差
    # 也就是允许偏差大的关节、允许偏差大的状态，惩罚更轻
    # 然后对所有关节求平均
    # 再做 exp(-x)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def joint_pos_limit(
        env: ManagerBasedRlEnv,
        limit: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        log_prefix: str = "Metrics/joint_pos_limit",
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    abs_error = torch.abs(diff_angle)
    excess = torch.clamp(abs_error - limit, min=0.0)
    penalty = torch.mean(torch.square(excess), dim=1)

    env.extras.setdefault("log", {})
    env.extras["log"][f"{log_prefix}/abs_error_mean"] = torch.mean(abs_error)
    env.extras["log"][f"{log_prefix}/excess_mean"] = torch.mean(excess)
    return penalty


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
