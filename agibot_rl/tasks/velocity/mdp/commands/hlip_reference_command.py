from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  wrap_to_pi,
)

HLIP_CLF_ERROR_SCALES = (
  0.20, 0.60,
  0.15, 0.50,
  0.08, 0.30,
  0.20, 1.00,
  0.20, 1.00,
  0.40, 1.00,
  0.08, 0.80,
  0.08, 0.80,
  0.08, 0.80,
  0.08, 0.80,
  0.08, 0.80,
  0.08, 0.80,
)
HLIP_CLF_Q_WEIGHTS = tuple(1.0 / (scale * scale) for scale in HLIP_CLF_ERROR_SCALES)
HLIP_CLF_R_WEIGHTS = (1.0,) * 12

def _bezier_deg(
  tau: torch.Tensor,
  control_points: torch.Tensor,
  degree: int,
) -> torch.Tensor:
  tau = torch.clamp(tau, 0.0, 1.0)
  coefs = torch.tensor(
    [math.comb(degree, idx) for idx in range(degree + 1)],
    dtype=control_points.dtype,
    device=control_points.device,
  )
  idx = torch.arange(degree + 1, device=control_points.device)
  terms = (
    control_points
    * coefs.unsqueeze(0)
    * ((1.0 - tau).unsqueeze(1) ** (degree - idx).unsqueeze(0))
    * (tau.unsqueeze(1) ** idx.unsqueeze(0))
  )
  return torch.sum(terms, dim=1)


def _bezier_deriv_deg(
  tau: torch.Tensor,
  duration: torch.Tensor,
  control_points: torch.Tensor,
  degree: int,
) -> torch.Tensor:
  tau = torch.clamp(tau, 0.0, 1.0)
  cp_diff = control_points[:, 1:] - control_points[:, :-1]
  coefs = torch.tensor(
    [math.comb(degree - 1, idx) for idx in range(degree)],
    dtype=control_points.dtype,
    device=control_points.device,
  )
  idx = torch.arange(degree, device=control_points.device)
  terms = (
    degree
    * cp_diff
    * coefs.unsqueeze(0)
    * ((1.0 - tau).unsqueeze(1) ** (degree - 1 - idx).unsqueeze(0))
    * (tau.unsqueeze(1) ** idx.unsqueeze(0))
  )
  return torch.sum(terms, dim=1) / torch.clamp(duration, min=1e-6)


def _euler_rates_to_omega(eul: torch.Tensor, eul_rates: torch.Tensor) -> torch.Tensor:
  phi, theta, psi = eul.unbind(-1)
  zeros = torch.zeros_like(theta)
  ones = torch.ones_like(theta)
  matrix = torch.stack(
    (
      torch.stack((torch.cos(theta) * torch.cos(psi), torch.sin(psi), zeros), dim=-1),
      torch.stack((-torch.cos(theta) * torch.sin(psi), torch.cos(psi), zeros), dim=-1),
      torch.stack((torch.sin(theta), zeros, ones), dim=-1),
    ),
    dim=-2,
  )
  return torch.einsum("bij,bj->bi", matrix, eul_rates)


class _ContinuousTimeClf:
  def __init__(
    self,
    n_outputs: int,
    dt: float,
    q_weights: tuple[float, ...],
    r_weights: tuple[float, ...],
    device: torch.device,
    yaw_idx: tuple[int, ...] = (),
  ):
    self.n_outputs = n_outputs
    self.dt = dt
    self.device = device
    self.yaw_idx = yaw_idx
    if len(q_weights) != 2 * n_outputs:
      raise ValueError(
        f"q_weights length must be {2 * n_outputs}, got {len(q_weights)}."
      )
    if len(r_weights) != n_outputs:
      raise ValueError(f"r_weights length must be {n_outputs}, got {len(r_weights)}.")

    q_np = np.diag(np.asarray(q_weights, dtype=np.float64))
    r_np = np.diag(np.asarray(r_weights, dtype=np.float64))
    a_block = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    b_block = np.array([[0.0], [1.0]], dtype=np.float64)
    a_full = np.kron(np.eye(n_outputs), a_block)
    b_full = np.kron(np.eye(n_outputs), b_block)
    p_np = solve_continuous_are(a_full, b_full, q_np, r_np)

    self.p = torch.as_tensor(p_np, dtype=torch.float32, device=device)
    self.lambda_max = torch.linalg.eigvalsh(self.p)[-1]
    self.norm_p = torch.linalg.norm(self.p, ord=2)
    self.norm_P = self.norm_p
    self.v_buffer: torch.Tensor | None = None
    self.last_y_err = torch.zeros(0, n_outputs, device=device)
    self.last_dy_err = torch.zeros(0, n_outputs, device=device)
    self.step_count = 0

  def reset(self, env_ids: torch.Tensor) -> None:
    if self.v_buffer is not None:
      self.v_buffer[env_ids] = 0.0

  def compute_v(
    self,
    y_act: torch.Tensor,
    y_ref: torch.Tensor,
    dy_act: torch.Tensor,
    dy_ref: torch.Tensor,
  ) -> torch.Tensor:
    if self.v_buffer is None or self.v_buffer.shape[0] != y_act.shape[0]:
      self.v_buffer = torch.zeros(y_act.shape[0], 3, device=y_act.device)
      self.step_count = 0

    y_err = y_act - y_ref
    dy_err = dy_act - dy_ref
    if self.yaw_idx:
      yaw_idx = torch.as_tensor(self.yaw_idx, device=y_act.device, dtype=torch.long)
      yaw_err = y_err[:, yaw_idx]
      y_err[:, yaw_idx] = (yaw_err + torch.pi) % (2.0 * torch.pi) - torch.pi
    self.last_y_err = y_err.detach()
    self.last_dy_err = dy_err.detach()
    eta = torch.zeros(y_act.shape[0], 2 * self.n_outputs, device=y_act.device)
    eta[:, 0::2] = y_err
    eta[:, 1::2] = dy_err
    v = torch.einsum("bi,ij,bj->b", eta, self.p, eta)

    self.v_buffer[:, 2] = self.v_buffer[:, 1]
    self.v_buffer[:, 1] = self.v_buffer[:, 0]
    self.v_buffer[:, 0] = v.detach()
    self.step_count += 1
    return v

  def compute_vdot(
    self,
    y_act: torch.Tensor,
    y_ref: torch.Tensor,
    dy_act: torch.Tensor,
    dy_ref: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    v_curr = self.compute_v(y_act, y_ref, dy_act, dy_ref)
    if self.step_count >= 3:
      vdot = (
        3.0 * self.v_buffer[:, 0]
        - 4.0 * self.v_buffer[:, 1]
        + self.v_buffer[:, 2]
      ) / (2.0 * self.dt)
    elif self.step_count == 2:
      vdot = (self.v_buffer[:, 0] - self.v_buffer[:, 1]) / self.dt
    else:
      vdot = torch.zeros_like(v_curr)
    return vdot, v_curr


class _HLIPFootPlacement:
  def __init__(
    self,
    device: torch.device,
    gravity: float,
    com_height: float,
    double_support_time: float,
    step_time: float,
    step_width: float,
  ):
    self.device = device
    self.gravity = gravity
    self.com_height = com_height
    self.double_support_time = double_support_time
    self.step_time = step_time
    self.step_width = step_width
    self.lambda_ = math.sqrt(self.gravity / self.com_height)
    single_support = self.step_time - self.double_support_time
    A_ss = torch.tensor(
      [[0.0, 1.0], [self.gravity / self.com_height, 0.0]],
      device=self.device,
    )
    A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=self.device)
    B = torch.tensor([-1.0, 0.0], device=self.device)
    self.A_s2s = torch.matrix_exp(A_ss * single_support) @ torch.matrix_exp(
      A_ds * self.double_support_time
    )
    self.B_s2s = torch.matrix_exp(A_ss * single_support) @ B

  def compute_orbit(
    self,
    command_b: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step_x = command_b[:, 0] * self.step_time
    eye = torch.eye(2, device=self.device).unsqueeze(0).expand(
      command_b.shape[0], -1, -1
    )
    x_init = torch.linalg.solve(
      eye - self.A_s2s,
      self.B_s2s.view(1, 2, 1) * step_x.view(-1, 1, 1),
    ).squeeze(-1)

    step_y_left = command_b[:, 1] * self.step_time + self.step_width
    step_y_right = command_b[:, 1] * self.step_time - self.step_width
    A_squared = self.A_s2s @ self.A_s2s
    B_term = self.A_s2s @ self.B_s2s
    y_left = torch.linalg.solve(
      eye - A_squared,
      B_term.view(1, 2, 1) * step_y_left.view(-1, 1, 1)
      + self.B_s2s.view(1, 2, 1) * step_y_right.view(-1, 1, 1),
    ).squeeze(-1)
    y_right = torch.linalg.solve(
      eye - A_squared,
      B_term.view(1, 2, 1) * step_y_right.view(-1, 1, 1)
      + self.B_s2s.view(1, 2, 1) * step_y_left.view(-1, 1, 1),
    ).squeeze(-1)
    return x_init, torch.stack((y_left, y_right), dim=1), step_x, torch.stack(
      (step_y_left, step_y_right), dim=1
    )

  def compute_com_trajectory(
    self,
    current_time: torch.Tensor,
    initial_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = initial_state[:, 0]
    v0 = initial_state[:, 1]
    lam = self.lambda_
    pos = x0 * torch.cosh(lam * current_time) + (v0 / lam) * torch.sinh(
      lam * current_time
    )
    vel = x0 * lam * torch.sinh(lam * current_time) + v0 * torch.cosh(
      lam * current_time
    )
    return pos, vel


class HLIPReferenceCommand(CommandTerm):
  cfg: HLIPReferenceCommandCfg

  def __init__(self, cfg: "HLIPReferenceCommandCfg", env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.foot_site_ids, self.foot_site_names = self.robot.find_sites(cfg.foot_site_names)
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
    self.prev_swing_foot_mask = torch.zeros_like(self.swing_foot_mask)
    self.swing_start_foot_pos_b = self._current_foot_pos_b()
    self.swing_start_foot_pos_l = self._current_foot_pos_b()
    self.stance_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.swing_idx = torch.ones_like(self.stance_idx)
    self.prev_stance_idx = torch.full_like(self.stance_idx, -1)
    foot_pos_w = self._current_foot_pos_w()
    foot_quat_w = self._current_foot_quat_w()
    self.stance_foot_pos_0 = foot_pos_w[:, 0, :].clone()
    self.stance_foot_ori_quat_0 = foot_quat_w[:, 0, :].clone()
    self.stance_foot_ori_0 = self._quat_to_rpy(self.stance_foot_ori_quat_0)
    self.hlip = _HLIPFootPlacement(
      device=self.device,
      gravity=cfg.hlip_gravity,
      com_height=cfg.hlip_com_height,
      double_support_time=cfg.hlip_double_support_time,
      step_time=cfg.reference_period,
      step_width=cfg.hlip_step_width,
    )
    self.ref_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
    self.ref_joint_ids_tensor = torch.empty(0, device=self.device, dtype=torch.long)
    self.n_outputs = 6 + 3 * len(self.foot_site_names)
    self.y_out = torch.zeros(self.num_envs, self.n_outputs, device=self.device)
    self.y_act = torch.zeros_like(self.y_out)
    self.dy_out = torch.zeros_like(self.y_out)
    self.dy_act = torch.zeros_like(self.y_out)
    self.v = torch.zeros(self.num_envs, device=self.device)
    self.vdot = torch.zeros_like(self.v)
    self.clf = _ContinuousTimeClf(
      n_outputs=self.n_outputs,
      dt=self._env.step_dt,
      q_weights=cfg.q_weights,
      r_weights=cfg.r_weights,
      device=self.device,
      yaw_idx=cfg.yaw_idx,
    )
    self.metrics["v"] = self.v
    self.metrics["vdot"] = self.vdot

  @property
  def command(self) -> torch.Tensor:
    return self.ref_swing_foot_pos_b.reshape(self.num_envs, -1)

  def __call__(self) -> torch.Tensor:
    return self.command

  def compute(self, dt: float) -> None:
    super().compute(dt)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.prev_swing_foot_mask[env_ids] = False
    self.swing_start_foot_pos_b[env_ids] = self._current_foot_pos_b()[env_ids]
    self.swing_start_foot_pos_l[env_ids] = self._current_foot_pos_l()[env_ids]
    foot_pos_w = self._current_foot_pos_w()
    foot_quat_w = self._current_foot_quat_w()
    self.stance_foot_pos_0[env_ids] = foot_pos_w[env_ids, self.stance_idx[env_ids], :]
    self.stance_foot_ori_quat_0[env_ids] = foot_quat_w[
      env_ids, self.stance_idx[env_ids], :
    ]
    self.stance_foot_ori_0[env_ids] = self._quat_to_rpy(
      self.stance_foot_ori_quat_0[env_ids]
    )
    self.prev_stance_idx[env_ids] = self.stance_idx[env_ids]
    self.clf.reset(env_ids)

  def _update_metrics(self) -> None:
    self.metrics["v"] = self.v
    self.metrics["vdot"] = self.vdot

  def _update_command(self) -> None:
    self._update_reference()

  def _phase_command(self):
    return self._env.command_manager.get_term(self.cfg.phase_command_name)

  def _current_foot_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.foot_site_ids_tensor, :]

  def _current_foot_quat_w(self) -> torch.Tensor:
    return self.robot.data.site_quat_w[:, self.foot_site_ids_tensor, :]

  def _quat_to_rpy(self, quat: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)), dim=-1)

  def _current_foot_rpy(self) -> torch.Tensor:
    foot_quat_w = self._current_foot_quat_w()
    return self._quat_to_rpy(foot_quat_w.reshape(-1, 4)).reshape(
      self.num_envs, len(self.foot_site_names), 3
    )

  def _current_foot_pos_b(self) -> torch.Tensor:
    foot_pos_w = self._current_foot_pos_w()
    root_pos_w = self.robot.data.root_link_pos_w.unsqueeze(1)
    root_quat_w = self.robot.data.root_link_quat_w.unsqueeze(1).expand(
      foot_pos_w.shape[0], foot_pos_w.shape[1], 4
    )
    return quat_apply_inverse(
      root_quat_w.reshape(-1, 4),
      (foot_pos_w - root_pos_w).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _current_foot_pos_l(self) -> torch.Tensor:
    foot_pos_w = self._current_foot_pos_w()
    stance_quat = self.stance_foot_ori_quat_0.unsqueeze(1).expand(
      foot_pos_w.shape[0], foot_pos_w.shape[1], 4
    )
    return quat_apply_inverse(
      stance_quat.reshape(-1, 4),
      (foot_pos_w - self.stance_foot_pos_0.unsqueeze(1)).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _current_foot_vel_b(self) -> torch.Tensor:
    foot_vel_w = self.robot.data.site_lin_vel_w[:, self.foot_site_ids_tensor, :]
    root_quat_w = self.robot.data.root_link_quat_w.unsqueeze(1).expand(
      foot_vel_w.shape[0], foot_vel_w.shape[1], 4
    )
    return quat_apply_inverse(
      root_quat_w.reshape(-1, 4),
      foot_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _current_foot_vel_l(self) -> torch.Tensor:
    foot_vel_w = self.robot.data.site_lin_vel_w[:, self.foot_site_ids_tensor, :]
    stance_quat = self.stance_foot_ori_quat_0.unsqueeze(1).expand(
      foot_vel_w.shape[0], foot_vel_w.shape[1], 4
    )
    return quat_apply_inverse(
      stance_quat.reshape(-1, 4),
      foot_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _current_foot_ang_vel_local(self) -> torch.Tensor:
    foot_ang_vel_w = self.robot.data.site_ang_vel_w[:, self.foot_site_ids_tensor, :]
    foot_quat_w = self._current_foot_quat_w()
    return quat_apply(
      quat_inv(foot_quat_w.reshape(-1, 4)),
      foot_ang_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)

  def _current_pelvis_rpy(self) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
    return torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)), dim=1)

  def _update_stance_initial_pose(self) -> None:
    foot_pos_w = self._current_foot_pos_w()
    foot_quat_w = self._current_foot_quat_w()
    env_ids = torch.arange(self.num_envs, device=self.device)
    reset_like = self._env.episode_length_buf <= 1
    changed = (self.stance_idx != self.prev_stance_idx) | reset_like
    self.stance_foot_pos_0 = torch.where(
      changed.unsqueeze(1),
      foot_pos_w[env_ids, self.stance_idx, :],
      self.stance_foot_pos_0,
    )
    self.stance_foot_ori_quat_0 = torch.where(
      changed.unsqueeze(1),
      foot_quat_w[env_ids, self.stance_idx, :],
      self.stance_foot_ori_quat_0,
    )
    self.stance_foot_ori_0 = torch.where(
      changed.unsqueeze(1),
      self._quat_to_rpy(self.stance_foot_ori_quat_0),
      self.stance_foot_ori_0,
    )
    self.prev_stance_idx = self.stance_idx.clone()

  def _pelvis_reference(
    self,
    command: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    tp = self.phase
    roll_ref = -0.05 * torch.sin(2.0 * torch.pi * tp)
    roll_ref = roll_ref + torch.clamp(torch.atan(command[:, 1] / self.cfg.hlip_gravity), -0.15, 0.15)
    roll_ref = roll_ref + torch.clamp(
      torch.atan((command[:, 0] * command[:, 2]) / self.cfg.hlip_gravity),
      -0.2,
      0.2,
    )
    pitch_ref = 0.02 * torch.sin(2.0 * torch.pi * tp)
    yaw_ref = self.stance_foot_ori_0[:, 2] + command[:, 2] * self.cur_swing_time
    pelvis_rpy_ref = torch.stack((roll_ref, pitch_ref, wrap_to_pi(yaw_ref)), dim=1)

    dphase_dt = 1.0 / torch.clamp(
      torch.full_like(tp, self.cfg.reference_period),
      min=1e-6,
    )
    eul_dot = torch.zeros_like(pelvis_rpy_ref)
    eul_dot[:, 0] = -0.05 * 2.0 * torch.pi * torch.cos(2.0 * torch.pi * tp) * dphase_dt
    eul_dot[:, 1] = 0.02 * 2.0 * torch.pi * torch.cos(2.0 * torch.pi * tp) * dphase_dt
    eul_dot[:, 2] = command[:, 2]
    return pelvis_rpy_ref, _euler_rates_to_omega(pelvis_rpy_ref, eul_dot)

  def _update_clf_state(
    self,
    command: torch.Tensor,
    phase: torch.Tensor,
    left_swing: torch.Tensor,
    foot_pos_b: torch.Tensor,
    ref_foot_pos_l: torch.Tensor,
    ref_foot_vel_l: torch.Tensor,
    ref_swing_foot_rpy: torch.Tensor,
    ref_swing_foot_ang_vel: torch.Tensor,
  ) -> None:
    foot_pos_l = self._current_foot_pos_l()
    foot_vel_l = self._current_foot_vel_l()
    foot_rpy = self._current_foot_rpy()
    foot_ang_vel = self._current_foot_ang_vel_local()
    root_height = torch.full(
      (self.num_envs,),
      self.cfg.hlip_com_height,
      device=self.device,
      dtype=foot_pos_b.dtype,
    )
    del foot_pos_b
    swing_indices = torch.where(left_swing, 0, 1)
    swing_foot_pos_l = foot_pos_l[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    swing_foot_vel_l = foot_vel_l[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    swing_foot_rpy = foot_rpy[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    swing_foot_ang_vel = foot_ang_vel[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    com_pos_l = quat_apply_inverse(
      self.stance_foot_ori_quat_0,
      self.robot.data.root_com_pos_w - self.stance_foot_pos_0,
    )
    com_vel_l = quat_apply_inverse(
      self.stance_foot_ori_quat_0,
      self.robot.data.root_com_vel_w[:, 0:3],
    )
    pelvis_rpy = self._current_pelvis_rpy()
    pelvis_rpy_ref, pelvis_rpy_rate_ref = self._pelvis_reference(command)

    cur_step_time = self.cur_swing_time
    x_ref, x_ref_dot = self.hlip.compute_com_trajectory(cur_step_time, self.hlip_x_init)
    y_state = torch.where(
      left_swing.unsqueeze(1),
      self.hlip_y_init[:, 0, :],
      self.hlip_y_init[:, 1, :],
    )
    y_ref, y_ref_dot = self.hlip.compute_com_trajectory(cur_step_time, y_state)
    root_ref = torch.stack((x_ref, y_ref, root_height), dim=1)
    root_ref_vel = torch.stack(
      (x_ref_dot, y_ref_dot, torch.zeros_like(x_ref_dot)),
      dim=1,
    )

    self.y_out = torch.cat(
      (root_ref, pelvis_rpy_ref, ref_foot_pos_l, ref_swing_foot_rpy),
      dim=1,
    )
    self.y_act = torch.cat(
      (com_pos_l, pelvis_rpy, swing_foot_pos_l, swing_foot_rpy),
      dim=1,
    )
    self.dy_out = torch.cat(
      (root_ref_vel, pelvis_rpy_rate_ref, ref_foot_vel_l, ref_swing_foot_ang_vel),
      dim=1,
    )
    self.dy_act = torch.cat(
      (com_vel_l, self.robot.data.root_link_ang_vel_b, swing_foot_vel_l, swing_foot_ang_vel),
      dim=1,
    )
    self.vdot, self.v = self.clf.compute_vdot(
      self.y_act,
      self.y_out,
      self.dy_act,
      self.dy_out,
    )
    self.metrics["v"] = self.v
    self.metrics["vdot"] = self.vdot

  def _update_reference(self) -> None:
    command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    foot_pos_b = self._current_foot_pos_b()
    foot_pos_l = self._current_foot_pos_l()
    phase_command = self._phase_command()
    self.phase = phase_command.phase
    self.phase_var = phase_command.phase_var
    self.cur_swing_time = phase_command.cur_swing_time
    self.stance_idx = phase_command.stance_idx
    self.swing_idx = phase_command.swing_idx
    self._update_stance_initial_pose()
    foot_pos_l = self._current_foot_pos_l()
    moving = torch.linalg.norm(command, dim=1) > self.cfg.reference_command_threshold
    left_swing = (self.swing_idx == 0) & moving
    right_swing = (self.swing_idx == 1) & moving
    swing_mask = torch.stack((left_swing, right_swing), dim=1)
    swing_start = swing_mask & ~self.prev_swing_foot_mask
    swing_start = swing_start | (self._env.episode_length_buf <= 1).unsqueeze(1)
    self.swing_start_foot_pos_b = torch.where(
      swing_start.unsqueeze(-1),
      foot_pos_b,
      self.swing_start_foot_pos_b,
    )
    self.swing_start_foot_pos_l = torch.where(
      swing_start.unsqueeze(-1),
      foot_pos_l,
      self.swing_start_foot_pos_l,
    )

    swing_phase = torch.stack((self.phase_var, self.phase_var), dim=1)
    swing_duration = torch.full_like(self.phase_var, 0.5 * self.cfg.reference_period)

    self.hlip_x_init, self.hlip_y_init, step_x, step_y_by_foot = self.hlip.compute_orbit(
      command
    )
    step_y = torch.where(left_swing, step_y_by_foot[:, 0], step_y_by_foot[:, 1])
    target_delta_xy = torch.stack((step_x, step_y), dim=1)
    target_delta_xy[:, 0] = torch.clamp(
      target_delta_xy[:, 0],
      min=self.cfg.swing_step_x_min,
      max=self.cfg.swing_step_x_max,
    )

    target_pos_l = self.swing_start_foot_pos_l.clone()
    target_pos_l[:, :, :2] = torch.where(
      swing_mask.unsqueeze(-1),
      target_delta_xy.unsqueeze(1),
      target_pos_l[:, :, :2],
    )

    horizontal_control = torch.tensor(
      (0.0, 0.0, 1.0, 1.0, 1.0),
      device=self.device,
      dtype=foot_pos_l.dtype,
    ).unsqueeze(0).expand(self.num_envs, -1)
    horizontal = torch.zeros_like(swing_phase)
    horizontal_dot = torch.zeros_like(swing_phase)
    for foot_idx in range(len(self.foot_site_names)):
      horizontal[:, foot_idx] = _bezier_deg(
        swing_phase[:, foot_idx],
        horizontal_control,
        4,
      )
      horizontal_dot[:, foot_idx] = _bezier_deriv_deg(
        swing_phase[:, foot_idx],
        swing_duration,
        horizontal_control,
        4,
      )

    x_ref_l = (
      self.swing_start_foot_pos_l[:, :, 0]
      + horizontal * (target_pos_l[:, :, 0] - self.swing_start_foot_pos_l[:, :, 0])
    )
    y_ref_l = (
      self.swing_start_foot_pos_l[:, :, 1]
      + horizontal * (target_pos_l[:, :, 1] - self.swing_start_foot_pos_l[:, :, 1])
    )
    x_ref_l_dot = horizontal_dot * (
      target_pos_l[:, :, 0] - self.swing_start_foot_pos_l[:, :, 0]
    )
    y_ref_l_dot = horizontal_dot * (
      target_pos_l[:, :, 1] - self.swing_start_foot_pos_l[:, :, 1]
    )
    z_init = self.swing_start_foot_pos_l[:, :, 2]
    z_land = target_pos_l[:, :, 2]
    z_max = torch.maximum(z_init, z_land) + self.cfg.swing_clearance
    z_ref = torch.zeros_like(z_init)
    z_ref_dot = torch.zeros_like(z_init)
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
      z_ref_dot[:, foot_idx] = _bezier_deriv_deg(
        swing_phase[:, foot_idx],
        swing_duration,
        control,
        6,
      )

    ref_pos_l = torch.stack((x_ref_l, y_ref_l, z_ref), dim=-1)
    ref_vel_l = torch.stack((x_ref_l_dot, y_ref_l_dot, z_ref_dot), dim=-1)
    stance_quat = self.stance_foot_ori_quat_0.unsqueeze(1).expand(
      self.num_envs, len(self.foot_site_names), 4
    )
    ref_pos_w = self.stance_foot_pos_0.unsqueeze(1) + quat_apply(
      stance_quat.reshape(-1, 4),
      ref_pos_l.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)
    root_quat = self.robot.data.root_link_quat_w.unsqueeze(1).expand_as(stance_quat)
    root_pos = self.robot.data.root_link_pos_w.unsqueeze(1)
    ref_pos_b = quat_apply_inverse(
      root_quat.reshape(-1, 4),
      (ref_pos_w - root_pos).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_site_names), 3)
    self.ref_swing_foot_pos_b = torch.where(
      swing_mask.unsqueeze(-1),
      ref_pos_b,
      foot_pos_b,
    )
    swing_indices = torch.where(left_swing, 0, 1)
    ref_swing_foot_pos_l = ref_pos_l[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    ref_swing_foot_vel_l = ref_vel_l[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    pelvis_rpy_ref, pelvis_rpy_rate_ref = self._pelvis_reference(command)
    ref_swing_foot_rpy = torch.zeros_like(pelvis_rpy_ref)
    ref_swing_foot_rpy[:, 2] = pelvis_rpy_ref[:, 2]
    ref_swing_foot_ang_vel = torch.zeros_like(ref_swing_foot_rpy)
    ref_swing_foot_ang_vel[:, 2] = pelvis_rpy_rate_ref[:, 2]
    self._update_clf_state(
      command,
      self.phase,
      left_swing,
      foot_pos_b,
      ref_swing_foot_pos_l,
      ref_swing_foot_vel_l,
      ref_swing_foot_rpy,
      ref_swing_foot_ang_vel,
    )
    self.swing_foot_phase = swing_phase
    self.swing_foot_mask = swing_mask
    self.prev_swing_foot_mask = swing_mask


@dataclass(kw_only=True)
class HLIPReferenceCommandCfg(CommandTermCfg):
  entity_name: str
  velocity_command_name: str = "twist"
  phase_command_name: str = "gait_phase"
  reference_period: float = 0.7
  reference_command_threshold: float = 0.1
  foot_site_names: tuple[str, ...]
  swing_clearance: float = 0.12
  swing_step_x_min: float = -0.20
  swing_step_x_max: float = 0.35
  hlip_gravity: float = 9.81
  hlip_com_height: float = 0.61
  hlip_double_support_time: float = 0.1
  hlip_step_width: float = 0.26
  q_weights: tuple[float, ...] = HLIP_CLF_Q_WEIGHTS
  r_weights: tuple[float, ...] = HLIP_CLF_R_WEIGHTS
  yaw_idx: tuple[int, ...] = (5, 11)

  def build(self, env) -> HLIPReferenceCommand:
    return HLIPReferenceCommand(self, env)
