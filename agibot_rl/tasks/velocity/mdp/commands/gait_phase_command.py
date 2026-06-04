from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


class GaitPhaseCommand(CommandTerm):
  cfg: GaitPhaseCommandCfg

  def __init__(self, cfg: "GaitPhaseCommandCfg", env):
    super().__init__(cfg, env)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.phase_var = torch.zeros_like(self.phase)
    self.cur_swing_time = torch.zeros_like(self.phase)
    self.stance_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.swing_idx = torch.ones_like(self.stance_idx)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.phase_var[env_ids] = 0.0
    self.cur_swing_time[env_ids] = 0.0
    self.stance_idx[env_ids] = 0
    self.swing_idx[env_ids] = 1
    self._command[env_ids] = 0.0

  def _update_metrics(self) -> None:
    self.metrics["phase"] = self.phase
    self.metrics["phase_var"] = self.phase_var

  def _update_command(self) -> None:
    elapsed = self._env.episode_length_buf.to(self.device) * self._env.step_dt
    self.phase = (elapsed / self.cfg.period) % 1.0
    first_half = self.phase < 0.5
    self.stance_idx = torch.where(
      first_half,
      torch.zeros_like(self.stance_idx),
      torch.ones_like(self.stance_idx),
    )
    self.swing_idx = 1 - self.stance_idx
    self.phase_var = torch.where(first_half, 2.0 * self.phase, 2.0 * self.phase - 1.0)
    self.cur_swing_time = self.phase_var * (0.5 * self.cfg.period)
    self._command = torch.stack(
      (
        torch.sin(2.0 * torch.pi * self.phase),
        torch.cos(2.0 * torch.pi * self.phase),
      ),
      dim=1,
    )


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  period: float

  def build(self, env) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
