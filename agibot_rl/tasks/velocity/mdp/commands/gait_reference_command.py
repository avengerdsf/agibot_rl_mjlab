from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .base_command import UniformVelocityCommand, UniformVelocityCommandCfg
from .hlip_reference_command import _HLIPFootPlacement, _bezier_deg


def _matched_values(
  names: list[str],
  patterns: dict[str, float],
  device: torch.device,
) -> torch.Tensor:
  values = torch.zeros(len(names), device=device, dtype=torch.float32)
  for idx, name in enumerate(names):
    for pattern, value in patterns.items():
      if re.fullmatch(pattern, name):
        values[idx] = value
        break
  return values

class GaitReferenceVelocityCommand(UniformVelocityCommand):
  cfg: GaitReferenceVelocityCommandCfg

  def __init__(self, cfg: GaitReferenceVelocityCommandCfg, env):
    super().__init__(cfg, env)

    self.ref_joint_ids, self.ref_joint_names = self.robot.find_joints(
      cfg.reference_joint_names
    )
    self.ref_joint_ids_tensor = torch.as_tensor(
      self.ref_joint_ids, device=self.device, dtype=torch.long
    )
    self.left_swing_offsets = _matched_values(
      self.ref_joint_names, cfg.left_swing_joint_offsets, self.device
    )
    self.right_swing_offsets = _matched_values(
      self.ref_joint_names, cfg.right_swing_joint_offsets, self.device
    )
    self.arm_swing_offsets = _matched_values(
      self.ref_joint_names, cfg.arm_swing_joint_offsets, self.device
    )
    self.ref_joint_pos = torch.zeros(
      self.num_envs, len(self.ref_joint_names), device=self.device
    )
    self.ref_joint_phase = torch.zeros(self.num_envs, device=self.device)
    if cfg.foot_site_names:
      self.foot_site_ids, self.foot_site_names = self.robot.find_sites(cfg.foot_site_names)
    else:
      self.foot_site_ids, self.foot_site_names = [], []
    self.foot_site_ids_tensor = torch.as_tensor(
      self.foot_site_ids, device=self.device, dtype=torch.long
    )
    self.ref_swing_foot_pos_b = torch.zeros(
      self.num_envs, len(self.foot_site_names), 3, device=self.device
    )
    self.swing_foot_phase = torch.zeros(
      self.num_envs, len(self.foot_site_names), device=self.device
    )
    self.swing_foot_mask = torch.zeros(
      self.num_envs, len(self.foot_site_names), device=self.device, dtype=torch.bool
    )
    self.hlip_x_init = torch.zeros(self.num_envs, 2, device=self.device)
    self.hlip_y_init = torch.zeros(self.num_envs, 2, 2, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.phase_var = torch.zeros_like(self.phase)
    self.cur_swing_time = torch.zeros_like(self.phase)
    self.stance_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.swing_idx = torch.ones_like(self.stance_idx)
    self.prev_swing_foot_mask = torch.zeros_like(self.swing_foot_mask)
    self.swing_start_foot_pos_b = self._current_foot_pos_b()
    self.hlip = _HLIPFootPlacement(
      device=self.device,
      gravity=cfg.hlip_gravity,
      com_height=cfg.hlip_com_height,
      double_support_time=cfg.hlip_double_support_time,
      step_time=cfg.reference_period,
      step_width=cfg.hlip_step_width,
    )
    self._update_gait_reference()

  def _current_foot_pos_b(self) -> torch.Tensor:
    if len(self.foot_site_names) == 0:
      return torch.zeros(self.num_envs, 0, 3, device=self.device)
    foot_pos_w = self.robot.data.site_pos_w[:, self.foot_site_ids_tensor, :]
    root_pos_w = self.robot.data.root_link_pos_w.unsqueeze(1)
    root_quat_w = self.robot.data.root_link_quat_w.unsqueeze(1).expand(
      foot_pos_w.shape[0], foot_pos_w.shape[1], 4
    )
    return quat_apply_inverse(
      root_quat_w.reshape(-1, 4),
      (foot_pos_w - root_pos_w).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _update_swing_foot_reference(
    self,
    phase: torch.Tensor,
    moving: torch.Tensor,
  ) -> None:
    if len(self.foot_site_names) != 2:
      return
    foot_pos_b = self._current_foot_pos_b()
    left_swing = (phase < 0.5) & moving
    right_swing = (phase >= 0.5) & moving
    swing_mask = torch.stack((left_swing, right_swing), dim=1)
    swing_start = swing_mask & ~self.prev_swing_foot_mask
    reset_like = self._env.episode_length_buf <= 1
    swing_start = swing_start | reset_like.unsqueeze(1)
    self.swing_start_foot_pos_b = torch.where(
      swing_start.unsqueeze(-1),
      foot_pos_b,
      self.swing_start_foot_pos_b,
    )

    left_phase = torch.clamp(phase / 0.5, 0.0, 1.0)
    right_phase = torch.clamp((phase - 0.5) / 0.5, 0.0, 1.0)
    swing_phase = torch.stack((left_phase, right_phase), dim=1)
    stance_indices = torch.where(left_swing, 1, 0)
    stance_foot_pos_b = foot_pos_b[
      torch.arange(self.num_envs, device=self.device),
      stance_indices,
    ]
    self.hlip_x_init, self.hlip_y_init, step_x, step_y_by_foot = self.hlip.compute_orbit(
      self.vel_command_b
    )
    step_y = torch.where(
      left_swing,
      step_y_by_foot[:, 0],
      step_y_by_foot[:, 1],
    )
    target_delta_xy = torch.stack((step_x, step_y), dim=1)
    target_delta_xy[:, 0] = torch.clamp(
      target_delta_xy[:, 0],
      min=self.cfg.swing_step_x_min,
      max=self.cfg.swing_step_x_max,
    )
    target_pos_b = self.swing_start_foot_pos_b.clone()
    target_pos_b[:, :, :2] = torch.where(
      swing_mask.unsqueeze(-1),
      stance_foot_pos_b[:, :2].unsqueeze(1) + target_delta_xy.unsqueeze(1),
      target_pos_b[:, :, :2],
    )
    x_ref = (
      (1.0 - swing_phase) * self.swing_start_foot_pos_b[:, :, 0]
      + swing_phase * target_pos_b[:, :, 0]
    )
    y_ref = self.swing_start_foot_pos_b[:, :, 1]
    z_init = self.swing_start_foot_pos_b[:, :, 2]
    z_land = target_pos_b[:, :, 2]
    z_max = torch.maximum(z_init, z_land) + self.cfg.swing_clearance
    z_ref = torch.zeros_like(z_init)
    for foot_idx in range(len(self.foot_site_names)):
      control = torch.stack(
        (
          z_init[:, foot_idx],
          z_init[:, foot_idx] + 0.2 * (z_max[:, foot_idx] - z_init[:, foot_idx]),
          z_init[:, foot_idx] + 0.6 * (z_max[:, foot_idx] - z_init[:, foot_idx]),
          z_max[:, foot_idx],
          z_land[:, foot_idx] + 0.5 * (z_max[:, foot_idx] - z_land[:, foot_idx]),
          z_land[:, foot_idx] + 0.05 * (z_max[:, foot_idx] - z_land[:, foot_idx]),
          z_land[:, foot_idx],
        ),
        dim=1,
      )
      z_ref[:, foot_idx] = _bezier_deg(swing_phase[:, foot_idx], control, 6)

    ref_pos = torch.stack((x_ref, y_ref, z_ref), dim=-1)
    self.ref_swing_foot_pos_b = torch.where(
      swing_mask.unsqueeze(-1),
      ref_pos,
      foot_pos_b,
    )
    self.swing_foot_phase = swing_phase
    self.swing_foot_mask = swing_mask
    self.prev_swing_foot_mask = swing_mask

  def _update_gait_reference(self) -> None:
    default_joint_pos = self.robot.data.default_joint_pos[:, self.ref_joint_ids_tensor]
    phase = ((self._env.episode_length_buf * self._env.step_dt) / self.cfg.reference_period) % 1.0
    sin_phase = torch.sin(2.0 * torch.pi * phase)
    left_swing = torch.clamp(sin_phase, min=0.0)
    right_swing = torch.clamp(-sin_phase, min=0.0)

    command_norm = torch.linalg.norm(self.vel_command_b, dim=1)
    moving = command_norm > self.cfg.reference_command_threshold

    offset = (
      left_swing.unsqueeze(1) * self.left_swing_offsets.unsqueeze(0)
      + right_swing.unsqueeze(1) * self.right_swing_offsets.unsqueeze(0)
      + sin_phase.unsqueeze(1) * self.arm_swing_offsets.unsqueeze(0)
    )
    offset = offset * moving.unsqueeze(1)

    self.ref_joint_pos = default_joint_pos + offset
    self.ref_joint_phase = phase
    self._update_swing_foot_reference(phase, moving)

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self._update_gait_reference()


@dataclass(kw_only=True)
class GaitReferenceVelocityCommandCfg(UniformVelocityCommandCfg):
  reference_joint_names: tuple[str, ...] = ()
  reference_period: float = 0.7
  reference_command_threshold: float = 0.1
  left_swing_joint_offsets: dict[str, float] = field(default_factory=dict)
  right_swing_joint_offsets: dict[str, float] = field(default_factory=dict)
  arm_swing_joint_offsets: dict[str, float] = field(default_factory=dict)
  foot_site_names: tuple[str, ...] = ()
  swing_clearance: float = 0.12
  swing_step_x_min: float = -0.20
  swing_step_x_max: float = 0.35
  hlip_gravity: float = 9.81
  hlip_com_height: float = 0.61
  hlip_double_support_time: float = 0.1
  hlip_step_width: float = 0.26

  def build(self, env) -> GaitReferenceVelocityCommand:
    return GaitReferenceVelocityCommand(self, env)

