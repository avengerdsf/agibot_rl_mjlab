from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat

from .hlip_reference_command import (
  HLIPReferenceCommand,
  HLIPReferenceCommandCfg,
  _body_omega_to_rpy_rates,
)


class FeedbackHLIPReferenceCommand(HLIPReferenceCommand):
  cfg: FeedbackHLIPReferenceCommandCfg

  def __init__(self, cfg: FeedbackHLIPReferenceCommandCfg, env):
    super().__init__(cfg, env)
    self.command_delta_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.velocity_feedback_gains = torch.tensor(
      cfg.velocity_feedback_gains,
      device=self.device,
      dtype=torch.float32,
    )
    self.roll_feedback_gains = torch.tensor(
      cfg.roll_feedback_gains,
      device=self.device,
      dtype=torch.float32,
    )
    self.pitch_feedback_gains = torch.tensor(
      cfg.pitch_feedback_gains,
      device=self.device,
      dtype=torch.float32,
    )
    self.max_command_delta = torch.tensor(
      cfg.max_command_delta,
      device=self.device,
      dtype=torch.float32,
    )
    self.command_limits = torch.tensor(
      cfg.command_limits,
      device=self.device,
      dtype=torch.float32,
    )
    self.metrics["feedback_delta_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_delta_y"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feedback_delta_yaw"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.command_delta_b[env_ids] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["feedback_delta_x"] = self.command_delta_b[:, 0]
    self.metrics["feedback_delta_y"] = self.command_delta_b[:, 1]
    self.metrics["feedback_delta_yaw"] = self.command_delta_b[:, 2]

  def _command_to_hlip_frame(
    self,
    command_b: torch.Tensor,
    root_quat_w: torch.Tensor,
    stance_foot_frame_w: torch.Tensor,
  ) -> torch.Tensor:
    command_feedback_b = self._apply_state_feedback(command_b)
    return HLIPReferenceCommand._command_to_hlip_frame(
      command_feedback_b,
      root_quat_w,
      stance_foot_frame_w,
    )

  def _apply_state_feedback(self, command_b: torch.Tensor) -> torch.Tensor:
    measured_velocity = torch.stack(
      (
        self.robot.data.root_link_lin_vel_b[:, 0],
        self.robot.data.root_link_lin_vel_b[:, 1],
        self.robot.data.root_link_ang_vel_b[:, 2],
      ),
      dim=1,
    )
    roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
    rpy = torch.stack((roll, pitch, yaw), dim=1)
    rpy_rate = _body_omega_to_rpy_rates(rpy, self.robot.data.root_link_ang_vel_b)
    velocity_error = command_b - measured_velocity
    target_delta = velocity_error * self.velocity_feedback_gains.unsqueeze(0)
    target_delta[:, 1] += (
      -rpy[:, 0] * self.roll_feedback_gains[0]
      - rpy_rate[:, 0] * self.roll_feedback_gains[1]
    )
    target_delta[:, 0] += (
      -rpy[:, 1] * self.pitch_feedback_gains[0]
      - rpy_rate[:, 1] * self.pitch_feedback_gains[1]
    )
    target_delta = torch.clamp(
      target_delta,
      min=-self.max_command_delta.unsqueeze(0),
      max=self.max_command_delta.unsqueeze(0),
    )
    self.command_delta_b = (
      (1.0 - self.cfg.feedback_alpha) * self.command_delta_b
      + self.cfg.feedback_alpha * target_delta
    )
    return torch.clamp(
      command_b + self.command_delta_b,
      min=self.command_limits[:, 0].unsqueeze(0),
      max=self.command_limits[:, 1].unsqueeze(0),
    )


@dataclass(kw_only=True)
class FeedbackHLIPReferenceCommandCfg(HLIPReferenceCommandCfg):
  velocity_feedback_gains: tuple[float, float, float] = (0.25, 0.15, 0.30)
  roll_feedback_gains: tuple[float, float] = (0.0, 0.0)
  pitch_feedback_gains: tuple[float, float] = (0.0, 0.0)
  max_command_delta: tuple[float, float, float] = (0.15, 0.10, 0.25)
  feedback_alpha: float = 0.2
  command_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.5, 0.5),
    (-0.5, 0.5),
    (-1.0, 1.0),
  )

  def build(self, env) -> FeedbackHLIPReferenceCommand:
    return FeedbackHLIPReferenceCommand(self, env)
