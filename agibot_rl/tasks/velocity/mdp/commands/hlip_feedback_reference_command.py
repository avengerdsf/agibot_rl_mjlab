from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

from .hlip_reference_command import (
  HLIPReferenceCommand,
  HLIPReferenceCommandCfg,
  _body_omega_to_rpy_rates,
)


class FeedbackHLIPReferenceCommand(HLIPReferenceCommand):
  cfg: FeedbackHLIPReferenceCommandCfg

  def __init__(self, cfg: FeedbackHLIPReferenceCommandCfg, env):
    super().__init__(cfg, env)
    self.yaw_ref_w = self.robot.data.heading_w.clone()
    self.yaw_reference_delta = torch.zeros(self.num_envs, device=self.device)
    self.yaw_rate_reference_delta = torch.zeros(self.num_envs, device=self.device)
    self.step_velocity_feedback_gains = torch.tensor(
      cfg.step_velocity_feedback_gains,
      device=self.device,
      dtype=torch.float32,
    )
    self.max_step_velocity_feedback = torch.tensor(
      cfg.max_step_velocity_feedback,
      device=self.device,
      dtype=torch.float32,
    )
    self.step_velocity_error_l = torch.zeros(self.num_envs, 2, device=self.device)
    self.step_velocity_feedback_delta = torch.zeros(self.num_envs, 2, device=self.device)
    self.metrics["feedback_delta_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_delta_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_yaw_rate_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_yaw_ref"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_step_velocity_error_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_step_velocity_error_y"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_step_delta_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_step_delta_y"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.yaw_ref_w[env_ids] = self.robot.data.heading_w[env_ids]
    self.yaw_reference_delta[env_ids] = 0.0
    self.yaw_rate_reference_delta[env_ids] = 0.0
    self.step_velocity_error_l[env_ids] = 0.0
    self.step_velocity_feedback_delta[env_ids] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["feedback_delta_yaw"] = self.yaw_reference_delta
    self.metrics["feedback_delta_yaw_rate"] = self.yaw_rate_reference_delta
    self.metrics["feedback_yaw_ref"] = self.yaw_ref_w
    self.metrics["feedback_step_velocity_error_x"] = self.step_velocity_error_l[:, 0]
    self.metrics["feedback_step_velocity_error_y"] = self.step_velocity_error_l[:, 1]
    self.metrics["feedback_step_delta_x"] = self.step_velocity_feedback_delta[:, 0]
    self.metrics["feedback_step_delta_y"] = self.step_velocity_feedback_delta[:, 1]

  def _command_to_hlip_frame(
    self,
    command_b: torch.Tensor,
    root_quat_w: torch.Tensor,
    stance_foot_frame_w: torch.Tensor,
  ) -> torch.Tensor:
    self._update_global_yaw_feedback(command_b)
    return HLIPReferenceCommand._command_to_hlip_frame(
      command_b,
      root_quat_w,
      stance_foot_frame_w,
    )

  def _pelvis_reference(
    self,
    command: torch.Tensor,
    stance_yaw_0: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    pelvis_rpy_ref, pelvis_rpy_rate_ref = super()._pelvis_reference(
      command,
      stance_yaw_0,
    )
    pelvis_rpy_ref[:, 2] = wrap_to_pi(
      pelvis_rpy_ref[:, 2] + self.yaw_reference_delta
    )
    pelvis_rpy_rate_ref[:, 2] = (
      pelvis_rpy_rate_ref[:, 2] + self.yaw_rate_reference_delta
    )
    return pelvis_rpy_ref, pelvis_rpy_rate_ref

  def _apply_step_velocity_feedback(
    self,
    target_delta_xy: torch.Tensor,
    swing_mask: torch.Tensor,
  ) -> torch.Tensor:
    _, x_ref_dot = self.hlip.compute_com_trajectory(
      self.cur_swing_time,
      self.hlip_x_init,
    )
    y_state = self.hlip_y_init[
      torch.arange(self.num_envs, device=self.device),
      self.stance_idx,
    ]
    _, y_ref_dot = self.hlip.compute_com_trajectory(self.cur_swing_time, y_state)
    ref_vel_l = torch.stack((x_ref_dot, y_ref_dot), dim=1)
    com_vel_l = self._world_to_hlip_frame(
      self.stance_foot_frame_w_0,
      self.robot.data.root_com_vel_w[:, 0:3],
    )
    velocity_error_l = com_vel_l[:, :2] - ref_vel_l
    feedback_delta = (
      velocity_error_l
      * self.step_velocity_feedback_gains.to(dtype=velocity_error_l.dtype).unsqueeze(0)
    )
    max_delta = self.max_step_velocity_feedback.to(
      dtype=feedback_delta.dtype
    ).unsqueeze(0)
    feedback_delta = torch.clamp(feedback_delta, min=-max_delta, max=max_delta)
    feedback_delta = torch.where(
      swing_mask.any(dim=1, keepdim=True),
      feedback_delta,
      torch.zeros_like(feedback_delta),
    )
    self.step_velocity_error_l = velocity_error_l
    self.step_velocity_feedback_delta = feedback_delta
    return target_delta_xy + feedback_delta

  def _update_global_yaw_feedback(self, command_b: torch.Tensor) -> None:
    actual_yaw = self.robot.data.heading_w
    reset_like = self._env.episode_length_buf <= 1
    self.yaw_ref_w = torch.where(reset_like, actual_yaw, self.yaw_ref_w)
    self.yaw_ref_w = wrap_to_pi(
      self.yaw_ref_w + command_b[:, 2] * self._env.step_dt
    )

    roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
    rpy = torch.stack((roll, pitch, yaw), dim=1)
    actual_yaw_rate = _body_omega_to_rpy_rates(
      rpy,
      self.robot.data.root_link_ang_vel_b,
    )[:, 2]
    yaw_error = wrap_to_pi(self.yaw_ref_w - actual_yaw)
    yaw_rate_error = command_b[:, 2] - actual_yaw_rate
    target_yaw_delta = self.cfg.yaw_feedback_gains[0] * yaw_error
    target_yaw_rate_delta = self.cfg.yaw_feedback_gains[1] * yaw_rate_error
    target_yaw_delta = torch.clamp(
      target_yaw_delta,
      min=-self.cfg.max_yaw_reference_delta,
      max=self.cfg.max_yaw_reference_delta,
    )
    target_yaw_rate_delta = torch.clamp(
      target_yaw_rate_delta,
      min=-self.cfg.max_yaw_rate_reference_delta,
      max=self.cfg.max_yaw_rate_reference_delta,
    )
    self.yaw_reference_delta = (
      (1.0 - self.cfg.feedback_alpha) * self.yaw_reference_delta
      + self.cfg.feedback_alpha * target_yaw_delta
    )
    self.yaw_rate_reference_delta = (
      (1.0 - self.cfg.feedback_alpha) * self.yaw_rate_reference_delta
      + self.cfg.feedback_alpha * target_yaw_rate_delta
    )
    self.metrics["feedback_yaw_error"] = yaw_error
    self.metrics["feedback_yaw_rate_error"] = yaw_rate_error


@dataclass(kw_only=True)
class FeedbackHLIPReferenceCommandCfg(HLIPReferenceCommandCfg):
  yaw_feedback_gains: tuple[float, float] = (1.0, 0.2)
  max_yaw_reference_delta: float = 0.25
  max_yaw_rate_reference_delta: float = 0.6
  step_velocity_feedback_gains: tuple[float, float] = (0.0, 0.0)
  max_step_velocity_feedback: tuple[float, float] = (0.0, 0.0)
  feedback_alpha: float = 0.2

  def build(self, env) -> FeedbackHLIPReferenceCommand:
    return FeedbackHLIPReferenceCommand(self, env)
