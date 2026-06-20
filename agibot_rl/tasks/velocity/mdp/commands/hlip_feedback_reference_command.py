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
    self.yaw_command_delta = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_delta_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_yaw_ref"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.yaw_ref_w[env_ids] = self.robot.data.heading_w[env_ids]
    self.yaw_command_delta[env_ids] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["feedback_delta_yaw"] = self.yaw_command_delta
    self.metrics["feedback_yaw_ref"] = self.yaw_ref_w

  def _command_to_hlip_frame(
    self,
    command_b: torch.Tensor,
    root_quat_w: torch.Tensor,
    stance_foot_frame_w: torch.Tensor,
  ) -> torch.Tensor:
    command_feedback_b = self._apply_global_yaw_feedback(command_b)
    return HLIPReferenceCommand._command_to_hlip_frame(
      command_feedback_b,
      root_quat_w,
      stance_foot_frame_w,
    )

  def _apply_global_yaw_feedback(self, command_b: torch.Tensor) -> torch.Tensor:
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
    target_delta = (
      self.cfg.yaw_feedback_gains[0] * yaw_error
      + self.cfg.yaw_feedback_gains[1] * yaw_rate_error
    )
    target_delta = torch.clamp(
      target_delta,
      min=-self.cfg.max_yaw_command_delta,
      max=self.cfg.max_yaw_command_delta,
    )
    self.yaw_command_delta = (
      (1.0 - self.cfg.feedback_alpha) * self.yaw_command_delta
      + self.cfg.feedback_alpha * target_delta
    )
    command_feedback_b = command_b.clone()
    command_feedback_b[:, 2] = torch.clamp(
      command_b[:, 2] + self.yaw_command_delta,
      min=self.cfg.yaw_command_limit[0],
      max=self.cfg.yaw_command_limit[1],
    )
    self.metrics["feedback_yaw_error"] = yaw_error
    return command_feedback_b


@dataclass(kw_only=True)
class FeedbackHLIPReferenceCommandCfg(HLIPReferenceCommandCfg):
  yaw_feedback_gains: tuple[float, float] = (1.0, 0.2)
  max_yaw_command_delta: float = 0.25
  feedback_alpha: float = 0.2
  yaw_command_limit: tuple[float, float] = (-1.0, 1.0)

  def build(self, env) -> FeedbackHLIPReferenceCommand:
    return FeedbackHLIPReferenceCommand(self, env)
