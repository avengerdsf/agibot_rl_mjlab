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
  wrap_to_pi,
  yaw_quat,
)

HLIP_CLF_Q_WEIGHTS = (
  25.0, 200.0,
  300.0, 50.0,
  400.0, 10.0,
  420.0, 20.0,
  200.0, 10.0,
  300.0, 10.0,
  1500.0, 125.0,
  1700.0, 125.0,
  3500.0, 100.0,
  30.0, 1.0,
  10.0, 1.0,
  400.0, 10.0,
  500.0, 10.0,
  40.0, 1.0,
  40.0, 1.0,
  100.0, 1.0,
  100.0, 1.0,
  50.0, 1.0,
  50.0, 1.0,
  30.0, 1.0,
  30.0, 1.0,
)
HLIP_CLF_R_WEIGHTS = (
  0.1, 0.1, 0.1,
  0.05, 0.05, 0.05,
  0.05, 0.05, 0.05,
  0.02, 0.02, 0.02,
  0.1,
  0.01, 0.01, 0.01,
  0.01, 0.01, 0.01,
  0.01, 0.01,
)


# Bezier 曲线的位置公式 
def _bezier_deg(
  tau: torch.Tensor,
  control_points: torch.Tensor,
  degree: int,
) -> torch.Tensor:
  # Bezier 曲线位置公式：
  #
  # B_n(tau) = sum_{i=0}^{n} C(n, i) * (1 - tau)^(n - i) * tau^i * P_i
  #
  # 其中：
  # n 表示 Bezier 曲线阶数，也就是 degree
  # tau 表示归一化时间相位，范围为 [0, 1]
  # P_i 表示第 i 个控制点
  # C(n, i) 表示组合数
  #
  # 本函数输出的是 Bezier 曲线在 tau 时刻的位置值
  tau = torch.clamp(tau, 0.0, 1.0)
  coefs = torch.tensor(
    [math.comb(degree, idx) for idx in range(degree + 1)],
    dtype=control_points.dtype,
    device=control_points.device,
  )
  idx = torch.arange(degree + 1, device=control_points.device)
  terms = control_points * coefs.unsqueeze(0) * ((1.0 - tau).unsqueeze(1) ** (degree - idx).unsqueeze(0)) * (tau.unsqueeze(1) ** idx.unsqueeze(0))
  return torch.sum(terms, dim=1)

# Bezier 曲线对真实时间的一阶导数公式
def _bezier_deriv_deg(
  tau: torch.Tensor,
  duration: torch.Tensor,
  control_points: torch.Tensor,
  degree: int,
) -> torch.Tensor:
  # Bezier 曲线对归一化相位 tau 的导数公式：
  #
  # dB_n(tau) / d tau
  # = n * sum_{i=0}^{n-1} C(n-1, i)
  #       * (1 - tau)^(n - 1 - i)
  #       * tau^i
  #       * (P_{i+1} - P_i)
  #
  # 如果真实时间为 t，且 tau = t / T，则：
  #
  # d tau / dt = 1 / T
  #
  # 因此 Bezier 曲线对真实时间的速度为：
  #
  # dB_n(tau) / dt = dB_n(tau) / d tau * 1 / T
  #
  # 也就是最后要除以 duration
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
  phi, theta, psi = eul.unbind(-1) # roll pitch yaw
  zeros = torch.zeros_like(theta)
  ones = torch.ones_like(theta)
  matrix = torch.stack(
    (
      torch.stack((torch.cos(theta) * torch.cos(psi), torch.sin(psi), zeros), dim=-1), # (B,3)
      torch.stack((-torch.cos(theta) * torch.sin(psi), torch.cos(psi), zeros), dim=-1),
      torch.stack((torch.sin(theta), zeros, ones), dim=-1),
    ),
    dim=-2,
  )# [B,3,3]
  return torch.einsum("bij,bj->bi", matrix, eul_rates)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
  return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
  lw, lx, ly, lz = lhs.unbind(-1)
  rw, rx, ry, rz = rhs.unbind(-1)
  return torch.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    dim=-1,
  )


class _ContinuousTimeClf:
  # 根据输出误差构造一个二次型李雅普诺夫函数，然后再利用差分近似计算李雅普诺夫函数的变化率
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

    # 李雅普诺夫函数里的二次型权重矩阵。
    # 把 numpy 里的 p_np 转成 PyTorch 张量
    self.p = torch.as_tensor(p_np, dtype=torch.float32, device=device)
    # eigvalsh 专门用于实对称矩阵 / Hermitian 矩阵的特征值，结果更稳定，而且输出的是实数从小到大排列.eigenvalues[-1]取最后一个，也就是最大特征值：
    self.lambda_max = torch.linalg.eigvalsh(self.p)[-1]
    # 求矩阵 P 的 2 范数，也叫谱范数（无穷范数 ord=float("inf"））
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
    # 检测环境数量匹配度创建缓存
    if self.v_buffer is None or self.v_buffer.shape[0] != y_act.shape[0]:
      self.v_buffer = torch.zeros(y_act.shape[0], 3, device=y_act.device)
      self.step_count = 0

    # 输出跟踪误差
    y_err = y_act - y_ref
    dy_err = dy_act - dy_ref
    if self.yaw_idx:
      yaw_idx = torch.as_tensor(self.yaw_idx, device=y_act.device, dtype=torch.long)
      yaw_err = y_err[:, yaw_idx]
      # 归一化角度误差
      y_err[:, yaw_idx] = (yaw_err + torch.pi) % (2.0 * torch.pi) - torch.pi
    # 得到的是一个新的 Tensor 对象，但是它通常和原来的 y 共享同一块数据内存，
    self.last_y_err = y_err.detach()
    self.last_dy_err = dy_err.detach()
    # 构造误差状态
    eta = torch.zeros(y_act.shape[0], 2 * self.n_outputs, device=y_act.device)
    eta[:, 0::2] = y_err
    eta[:, 1::2] = dy_err
    # 批量计算 CLF 的二次型
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
      # 二阶后向差分公式
      vdot = (
        3.0 * self.v_buffer[:, 0]
        - 4.0 * self.v_buffer[:, 1]
        + self.v_buffer[:, 2]
      ) / (2.0 * self.dt)
    elif self.step_count == 2:
      # 一阶后向差分
      vdot = (self.v_buffer[:, 0] - self.v_buffer[:, 1]) / self.dt
    else:
      # 第一帧没有历史数据，不能算变化率，所以直接给 0
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
    # 单支撑相位矩阵 单支撑阶段的线性倒立摆动力学：
    A_ss = torch.tensor(
      [[0.0, 1.0], [self.gravity / self.com_height, 0.0]],
      device=self.device,
    )
    # 双支撑相位矩阵 表示双支撑阶段近似为匀速运动：
    A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=self.device)
    # 是落足点输入对状态的影响
    B = torch.tensor([-1.0, 0.0], device=self.device)
    # torch.matrix_exp(A_ss * single_support) 连续系统的精确离散化
    # 从一个支撑相位到下一个支撑相位的整体状态转移
    self.A_s2s = torch.matrix_exp(A_ss * single_support) @ torch.matrix_exp(
      A_ds * self.double_support_time
    )
    # 是落足点输入经过单支撑传播后的影响
    self.B_s2s = torch.matrix_exp(A_ss * single_support) @ B

  def compute_orbit(
    self,
    command_b: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step_x = command_b[:, 0] * self.step_time # 计算前向步长 step_x
    # 构造单位矩阵 eye [4096,2,2]
    eye = torch.eye(2, device=self.device).unsqueeze(0).expand(
      command_b.shape[0], -1, -1
    )
    # 求 x 方向 HLIP 周期轨道的初始状态：
    x_init = torch.linalg.solve(
      eye - self.A_s2s,
      self.B_s2s.view(1, 2, 1) * step_x.view(-1, 1, 1),
    ).squeeze(-1)

    # 求步宽
    step_y_left = command_b[:, 1] * self.step_time + self.step_width
    step_y_right = command_b[:, 1] * self.step_time - self.step_width
    # 两步转移矩阵
    A_squared = self.A_s2s @ self.A_s2s
    # 输入的一步转移矩阵
    B_term = self.A_s2s @ self.B_s2s
    # 求解轨道的初始状态，也就是经过两步传播回到相同的状态
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
    # 倒立摆模型解析解
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
    self.foot_body_ids, self.foot_body_names = self.robot.find_bodies(cfg.foot_body_names)
    self.foot_body_ids_tensor = torch.as_tensor(
      self.foot_body_ids, device=self.device, dtype=torch.long
    )
    self.ref_swing_foot_pos_b = torch.zeros(
      self.num_envs, len(self.foot_body_names), 3, device=self.device
    )
    self.swing_foot_phase = torch.zeros(
      self.num_envs, len(self.foot_body_names), device=self.device
    )
    self.swing_foot_mask = torch.zeros(
      self.num_envs, len(self.foot_body_names), device=self.device, dtype=torch.bool
    )
    self.hlip_x_init = torch.zeros(self.num_envs, 2, device=self.device)
    self.hlip_y_init = torch.zeros(self.num_envs, 2, 2, device=self.device)
    self.upper_body_joint_ids, self.upper_body_joint_names = self.robot.find_joints(
      cfg.upper_body_joint_names
    )
    upper_body_order = (
      "lumbar_yaw",
      "left_shoulder_pitch",
      "right_shoulder_pitch",
      "left_shoulder_roll",
      "right_shoulder_roll",
      "left_shoulder_yaw",
      "right_shoulder_yaw",
      "left_elbow_pitch",
      "right_elbow_pitch",
    )
    ordered_joint_ids = []
    ordered_joint_names = []
    for joint_key in upper_body_order:
      matches = [
        (joint_id, joint_name)
        for joint_id, joint_name in zip(self.upper_body_joint_ids, self.upper_body_joint_names)
        if joint_key in joint_name
      ]
      if len(matches) == 1:
        ordered_joint_ids.append(matches[0][0])
        ordered_joint_names.append(matches[0][1])
    self.upper_body_joint_ids = ordered_joint_ids
    self.upper_body_joint_names = ordered_joint_names
    if len(self.upper_body_joint_ids) != 9:
      raise ValueError(
        "HLIP upper-body reference expects 9 joints in X1 order, "
        f"got {len(self.upper_body_joint_ids)}: {self.upper_body_joint_names}."
      )
    self.upper_body_joint_ids_tensor = torch.as_tensor(
      self.upper_body_joint_ids, device=self.device, dtype=torch.long
    )
    self.foot_heading_axis_b = torch.tensor((0.0, 0.0, 1.0), device=self.device)
    self.prev_swing_foot_mask = torch.zeros_like(self.swing_foot_mask)
    self.prev_foot_contact = torch.zeros_like(self.swing_foot_mask)
    self.swing_start_foot_pos_b = self._current_foot_pos_b()
    self.swing_start_foot_pos_l = self._current_foot_pos_b()
    self.step_target_delta_xy = torch.zeros(
      self.num_envs, len(self.foot_body_names), 2, device=self.device
    )
    self.last_landing_actual_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.last_landing_target_xy = torch.zeros_like(self.last_landing_actual_xy)
    self.last_landing_error_xy = torch.zeros_like(self.last_landing_actual_xy)
    self.last_landing_valid = torch.zeros(self.num_envs, device=self.device)
    self.stance_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.swing_idx = torch.ones_like(self.stance_idx)
    self.prev_stance_idx = torch.full_like(self.stance_idx, -1)
    self.pelvis_rpy = self._current_pelvis_rpy()
    self.prev_pelvis_rpy = self.pelvis_rpy.clone()
    self.pelvis_rpy_rate = torch.zeros_like(self.pelvis_rpy)
    foot_pos_w = self._current_foot_pos_w()
    foot_quat_w = self._current_foot_quat_w()
    self.foot_rpy = self._foot_quat_to_rpy(foot_quat_w)
    self.prev_foot_rpy = self.foot_rpy.clone()
    self.foot_rpy_rate = torch.zeros_like(self.foot_rpy)
    self.stance_foot_pos_0 = foot_pos_w[:, 0, :].clone()
    self.stance_foot_ori_quat_0 = foot_quat_w[:, 0, :].clone()
    self.stance_foot_ori_0 = self._foot_quat_to_rpy(self.stance_foot_ori_quat_0)
    self.stance_foot_pos = self.stance_foot_pos_0.clone()
    self.stance_foot_ori = self.stance_foot_ori_0.clone()
    self.stance_foot_vel = torch.zeros(self.num_envs, 3, device=self.device)
    self.stance_foot_rpy_rate = torch.zeros_like(self.stance_foot_vel)
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
    # COM xyz, pelvis rpy, swing foot xyz/rpy, then target-project upper-body joints.
    self.n_outputs = 6 + 3 * len(self.foot_body_names) + len(self.upper_body_joint_ids)
    self.output_names = (
      "com_x",
      "com_y",
      "com_z",
      "pelvis_roll",
      "pelvis_pitch",
      "pelvis_yaw",
      "swing_foot_x",
      "swing_foot_y",
      "swing_foot_z",
      "swing_foot_roll",
      "swing_foot_pitch",
      "swing_foot_yaw",
      *self.upper_body_joint_names,
    )
    self.y_out = torch.zeros(self.num_envs, self.n_outputs, device=self.device)
    self.y_act = torch.zeros_like(self.y_out)
    self.dy_out = torch.zeros_like(self.y_out)
    self.dy_act = torch.zeros_like(self.y_out)
    self.upper_body_joint_pos = torch.zeros(
      self.num_envs, len(self.upper_body_joint_ids), device=self.device
    )
    self.upper_body_joint_vel = torch.zeros_like(self.upper_body_joint_pos)
    self.prev_upper_body_joint_pos = self.robot.data.joint_pos[
      :, self.upper_body_joint_ids_tensor
    ].clone()
    self.upper_body_joint_pos_diff_vel = torch.zeros_like(self.upper_body_joint_pos)
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
    foot_rpy = self._current_foot_rpy()
    pelvis_rpy = self._current_pelvis_rpy()
    self.foot_rpy[env_ids] = foot_rpy[env_ids]
    self.prev_foot_rpy[env_ids] = foot_rpy[env_ids]
    self.foot_rpy_rate[env_ids] = 0.0
    self.pelvis_rpy[env_ids] = pelvis_rpy[env_ids]
    self.prev_pelvis_rpy[env_ids] = pelvis_rpy[env_ids]
    self.pelvis_rpy_rate[env_ids] = 0.0
    self.prev_swing_foot_mask[env_ids] = False
    self.prev_foot_contact[env_ids] = self._current_foot_contact()[env_ids]
    self.swing_start_foot_pos_b[env_ids] = self._current_foot_pos_b()[env_ids]
    self.swing_start_foot_pos_l[env_ids] = self._current_foot_pos_l()[env_ids]
    self.step_target_delta_xy[env_ids] = 0.0
    self.last_landing_actual_xy[env_ids] = 0.0
    self.last_landing_target_xy[env_ids] = 0.0
    self.last_landing_error_xy[env_ids] = 0.0
    self.last_landing_valid[env_ids] = 0.0
    foot_pos_w = self._current_foot_pos_w()
    foot_quat_w = self._current_foot_quat_w()
    self.stance_foot_pos_0[env_ids] = foot_pos_w[env_ids, self.stance_idx[env_ids], :]
    self.stance_foot_ori_quat_0[env_ids] = foot_quat_w[
      env_ids, self.stance_idx[env_ids], :
    ]
    self.stance_foot_ori_0[env_ids] = self._foot_quat_to_rpy(
      self.stance_foot_ori_quat_0[env_ids]
    )
    current_upper_body_joint_pos = self.robot.data.joint_pos[
      :, self.upper_body_joint_ids_tensor
    ]
    self.upper_body_joint_pos[env_ids] = current_upper_body_joint_pos[env_ids]
    self.upper_body_joint_vel[env_ids] = self.robot.data.joint_vel[
      :, self.upper_body_joint_ids_tensor
    ][env_ids]
    self.prev_upper_body_joint_pos[env_ids] = current_upper_body_joint_pos[env_ids]
    self.upper_body_joint_pos_diff_vel[env_ids] = 0.0
    self.prev_stance_idx[env_ids] = self.stance_idx[env_ids]
    self.clf.reset(env_ids)

  def _update_metrics(self) -> None:
    self.metrics["v"] = self.v
    self.metrics["vdot"] = self.vdot

  def _update_command(self) -> None:
    self._update_reference()

  def _current_foot_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.foot_body_ids_tensor, :]

  def _current_foot_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.foot_body_ids_tensor, :]

  def _quat_to_rpy(self, quat: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)), dim=-1)

  def _foot_heading_yaw(self, foot_quat_w: torch.Tensor) -> torch.Tensor:
    forward_axis = self.foot_heading_axis_b.to(dtype=foot_quat_w.dtype).expand(
      foot_quat_w.reshape(-1, 4).shape[0], 3
    )
    heading_vec = quat_apply(foot_quat_w.reshape(-1, 4), forward_axis).reshape(foot_quat_w.shape[:-1] + (3,))
    return torch.atan2(heading_vec[..., 1], heading_vec[..., 0])

  def _foot_quat_to_rpy(self, foot_quat_w: torch.Tensor) -> torch.Tensor:
    foot_rpy = self._quat_to_rpy(foot_quat_w.reshape(-1, 4)).reshape(foot_quat_w.shape[:-1] + (3,))
    foot_rpy[..., 2] = self._foot_heading_yaw(foot_quat_w)
    return foot_rpy

  def _current_foot_rpy(self) -> torch.Tensor:
    foot_quat_w = self._current_foot_quat_w()
    return self._foot_quat_to_rpy(foot_quat_w)

  def _rpy_rate_from_previous(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return wrap_to_pi(current - previous) / max(self._env.step_dt, 1e-6)

  def _current_foot_contact(self) -> torch.Tensor:
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    assert sensor.data.found is not None
    contact = sensor.data.found > 0
    num_feet = len(self.foot_body_names)
    if contact.shape[1] == num_feet:
      return contact
    if contact.shape[1] % num_feet != 0:
      raise RuntimeError(
        f"HLIP landing metric expected contact count divisible by num_feet={num_feet}, "
        f"got {contact.shape[1]}."
      )
    geoms_per_foot = contact.shape[1] // num_feet
    return contact.reshape(contact.shape[0], num_feet, geoms_per_foot).any(dim=-1)

  def get_not_flight_envs(self) -> torch.Tensor:
    contact = self._current_foot_contact()
    env_ids = torch.arange(self.num_envs, device=self.device)
    return contact[env_ids, self.stance_idx].float()

  def _current_foot_pos_b(self) -> torch.Tensor:
    foot_pos_w = self._current_foot_pos_w()
    root_pos_w = self.robot.data.root_link_pos_w.unsqueeze(1)
    root_quat_w = self.robot.data.root_link_quat_w.unsqueeze(1).expand(
      foot_pos_w.shape[0], foot_pos_w.shape[1], 4
    )
    return quat_apply_inverse(
      root_quat_w.reshape(-1, 4),
      (foot_pos_w - root_pos_w).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)

  def _current_foot_pos_l(self) -> torch.Tensor:
    foot_pos_w = self._current_foot_pos_w()
    stance_quat = self.stance_foot_ori_quat_0.unsqueeze(1).expand(
      foot_pos_w.shape[0], foot_pos_w.shape[1], 4
    )
    return quat_apply_inverse(
      stance_quat.reshape(-1, 4),
      (foot_pos_w - self.stance_foot_pos_0.unsqueeze(1)).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)

  def _current_foot_vel_b(self) -> torch.Tensor:
    foot_vel_w = self.robot.data.body_link_lin_vel_w[:, self.foot_body_ids_tensor, :]
    root_quat_w = self.robot.data.root_link_quat_w.unsqueeze(1).expand(
      foot_vel_w.shape[0], foot_vel_w.shape[1], 4
    )
    return quat_apply_inverse(
      root_quat_w.reshape(-1, 4),
      foot_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)

  def _current_foot_vel_l(self) -> torch.Tensor:
    foot_vel_w = self.robot.data.body_link_lin_vel_w[:, self.foot_body_ids_tensor, :]
    stance_quat = self.stance_foot_ori_quat_0.unsqueeze(1).expand(
      foot_vel_w.shape[0], foot_vel_w.shape[1], 4
    )
    return quat_apply_inverse(
      stance_quat.reshape(-1, 4),
      foot_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)

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
      self._foot_quat_to_rpy(self.stance_foot_ori_quat_0),
      self.stance_foot_ori_0,
    )
    self.prev_stance_idx = self.stance_idx.clone()

  def _update_stance_state(self) -> None:
    env_ids = torch.arange(self.num_envs, device=self.device)
    foot_pos_w = self._current_foot_pos_w()
    foot_rpy = self._current_foot_rpy()
    foot_vel_w = self.robot.data.body_link_lin_vel_w[:, self.foot_body_ids_tensor, :]
    foot_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self.foot_body_ids_tensor, :]
    foot_ang_vel = quat_apply_inverse(
      self._current_foot_quat_w().reshape(-1, 4),
      foot_ang_vel_w.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)
    pelvis_rpy = self._current_pelvis_rpy()
    self.foot_rpy = foot_rpy
    self.foot_rpy_rate = foot_ang_vel
    self.pelvis_rpy = pelvis_rpy
    self.pelvis_rpy_rate = self.robot.data.root_link_ang_vel_b
    self.stance_foot_pos = foot_pos_w[env_ids, self.stance_idx, :]
    self.stance_foot_ori = foot_rpy[env_ids, self.stance_idx, :]
    self.stance_foot_vel = foot_vel_w[env_ids, self.stance_idx, :]
    self.stance_foot_rpy_rate = foot_ang_vel[env_ids, self.stance_idx, :]

  def _update_phase_state(self) -> None:
    elapsed = self._env.episode_length_buf.to(self.device) * self._env.step_dt
    self.phase = (elapsed / self.cfg.reference_period) % 1.0
    first_half = self.phase < 0.5
    self.stance_idx = torch.where(
      first_half,
      torch.zeros_like(self.stance_idx),
      torch.ones_like(self.stance_idx),
    )
    self.swing_idx = 1 - self.stance_idx
    self.phase_var = torch.where(
      first_half,
      2.0 * self.phase,
      2.0 * self.phase - 1.0,
    )
    self.cur_swing_time = self.phase_var * (0.5 * self.cfg.reference_period)

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
    yaw_ref = command[:, 2] * self.cur_swing_time
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

  def _upper_body_reference(
    self,
    command: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    forward_vel = command[:, 0]
    phase = 2.0 * torch.pi * self.phase
    sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
    elb0 = self.cfg.elbow_ref
    waist_yaw0 = self.cfg.waist_yaw_ref
    amp = torch.zeros(
      self.num_envs,
      len(self.upper_body_joint_names),
      device=self.device,
      dtype=forward_vel.dtype,
    )
    sign = torch.ones_like(amp)
    offset = torch.zeros_like(amp)
    for joint_idx, joint_name in enumerate(self.upper_body_joint_names):
      if joint_name == "lumbar_yaw_joint":
        amp[:, joint_idx] = waist_yaw0
        offset[:, joint_idx] = torch.pi
      elif "shoulder_pitch" in joint_name:
        amp[:, joint_idx] = sh_pitch0 * forward_vel
        sign[:, joint_idx] = 1.0 if joint_name.startswith("left_") else -1.0
        offset[:, joint_idx] = torch.pi / 2.0
      elif "shoulder_roll" in joint_name:
        amp[:, joint_idx] = sh_roll0
        sign[:, joint_idx] = 1.0 if joint_name.startswith("left_") else -1.0
        offset[:, joint_idx] = torch.pi / 2.0
      elif "shoulder_yaw" in joint_name:
        amp[:, joint_idx] = sh_yaw0
        sign[:, joint_idx] = 1.0 if joint_name.startswith("left_") else -1.0
      elif "elbow_pitch" in joint_name:
        amp[:, joint_idx] = elb0 * forward_vel
        sign[:, joint_idx] = 1.0 if joint_name.startswith("left_") else -1.0
        offset[:, joint_idx] = torch.pi / 2.0
    default_joint_pos = self.robot.data.default_joint_pos[
      :, self.upper_body_joint_ids_tensor
    ]
    ref = (
      amp
      * sign
      * torch.sin(phase.unsqueeze(1) + offset)
      + default_joint_pos
    )
    dphase_dt = 2.0 * torch.pi / torch.clamp(
      torch.full_like(forward_vel, self.cfg.reference_period),
      min=1e-6,
    )
    ref_dot = (
      amp
      * sign
      * torch.cos(phase.unsqueeze(1) + offset)
      * dphase_dt.unsqueeze(1)
    )
    return ref, ref_dot

  def _update_clf_state(
    self,
    command: torch.Tensor,
    phase: torch.Tensor,
    left_swing: torch.Tensor,
    foot_pos_b: torch.Tensor,
    ref_foot_pos_l: torch.Tensor,
    ref_foot_vel_l: torch.Tensor,
    ref_swing_foot_rpy: torch.Tensor,
    ref_swing_foot_rpy_rate: torch.Tensor,
    ref_upper_body_joint_pos: torch.Tensor,
    ref_upper_body_joint_vel: torch.Tensor,
  ) -> None:
    foot_pos_l = self._current_foot_pos_l()
    foot_vel_l = self._current_foot_vel_l()
    foot_rpy = self.foot_rpy
    foot_rpy_rate = self.foot_rpy_rate
    root_height = torch.full(
      (self.num_envs,),
      self.cfg.hlip_com_height,
      device=self.device,
      dtype=foot_pos_b.dtype,
    )
    del foot_pos_b
    foot_quat_w = self._current_foot_quat_w()
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
    ].clone()
    swing_foot_quat_w = foot_quat_w[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    swing_foot_rpy[:, :2] = self._quat_to_rpy(
      _quat_mul(_quat_conjugate(self.stance_foot_ori_quat_0), swing_foot_quat_w)
    )[:, :2]
    swing_foot_rpy_rate = foot_rpy_rate[
      torch.arange(self.num_envs, device=self.device),
      swing_indices,
    ]
    stance_yaw_quat_0 = yaw_quat(self.stance_foot_ori_quat_0)
    com_pos_l = quat_apply_inverse(
      stance_yaw_quat_0,
      self.robot.data.root_com_pos_w - self.stance_foot_pos_0,
    )
    com_vel_l = quat_apply_inverse(
      stance_yaw_quat_0,
      self.robot.data.root_com_vel_w[:, 0:3],
    )
    stance_yaw_0 = self.stance_foot_ori_0[:, 2]
    swing_foot_rpy[:, 2] = wrap_to_pi(swing_foot_rpy[:, 2] - stance_yaw_0)
    pelvis_rpy = self.pelvis_rpy.clone()
    pelvis_rpy[:, 2] = wrap_to_pi(pelvis_rpy[:, 2] - stance_yaw_0)
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
    delta_yaw = command[:, 2] * cur_step_time
    cos_yaw = torch.cos(delta_yaw)
    sin_yaw = torch.sin(delta_yaw)
    root_ref_xy = root_ref[:, :2].clone()
    root_ref_vel_xy = root_ref_vel[:, :2].clone()
    root_ref[:, 0] = cos_yaw * root_ref_xy[:, 0] - sin_yaw * root_ref_xy[:, 1]
    root_ref[:, 1] = sin_yaw * root_ref_xy[:, 0] + cos_yaw * root_ref_xy[:, 1]
    root_ref_vel[:, 0] = cos_yaw * root_ref_vel_xy[:, 0] - sin_yaw * root_ref_vel_xy[:, 1]
    root_ref_vel[:, 1] = sin_yaw * root_ref_vel_xy[:, 0] + cos_yaw * root_ref_vel_xy[:, 1]
    current_upper_body_joint_pos = self.robot.data.joint_pos[
      :, self.upper_body_joint_ids_tensor
    ]
    current_upper_body_joint_vel = self.robot.data.joint_vel[
      :, self.upper_body_joint_ids_tensor
    ]
    self.upper_body_joint_pos_diff_vel = (
      wrap_to_pi(current_upper_body_joint_pos - self.prev_upper_body_joint_pos)
      / max(self._env.step_dt, 1e-6)
    )
    self.upper_body_joint_pos = current_upper_body_joint_pos
    self.upper_body_joint_vel = current_upper_body_joint_vel
    self.prev_upper_body_joint_pos = current_upper_body_joint_pos.clone()

    self.y_out = torch.cat(
      (
        root_ref,
        pelvis_rpy_ref,
        ref_foot_pos_l,
        ref_swing_foot_rpy,
        ref_upper_body_joint_pos,
      ),
      dim=1,
    )
    self.y_act = torch.cat(
      (
        com_pos_l,
        pelvis_rpy,
        swing_foot_pos_l,
        swing_foot_rpy,
        current_upper_body_joint_pos,
      ),
      dim=1,
    )
    self.dy_out = torch.cat(
      (
        root_ref_vel,
        pelvis_rpy_rate_ref,
        ref_foot_vel_l,
        ref_swing_foot_rpy_rate,
        ref_upper_body_joint_vel,
      ),
      dim=1,
    )
    self.dy_act = torch.cat(
      (
        com_vel_l,
        self.pelvis_rpy_rate,
        swing_foot_vel_l,
        swing_foot_rpy_rate,
        current_upper_body_joint_vel,
      ),
      dim=1,
    )
    self.vdot, self.v = self.clf.compute_vdot(
      self.y_act,
      self.y_out,
      self.dy_act,
      self.dy_out,
    )
    com_error = com_pos_l - root_ref
    self.metrics["com_actual_x"] = com_pos_l[:, 0].clone()
    self.metrics["com_actual_y"] = com_pos_l[:, 1].clone()
    self.metrics["com_actual_z"] = com_pos_l[:, 2].clone()
    self.metrics["com_ref_x"] = root_ref[:, 0].clone()
    self.metrics["com_ref_y"] = root_ref[:, 1].clone()
    self.metrics["com_ref_z"] = root_ref[:, 2].clone()
    self.metrics["com_error_x"] = com_error[:, 0].clone()
    self.metrics["com_error_y"] = com_error[:, 1].clone()
    self.metrics["com_error_z"] = com_error[:, 2].clone()
    self.metrics["com_error_xy_norm"] = torch.linalg.norm(com_error[:, :2], dim=1)
    self.metrics["v"] = self.v
    self.metrics["vdot"] = self.vdot

  def _update_reference(self) -> None:
    command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    foot_pos_b = self._current_foot_pos_b()
    foot_pos_l = self._current_foot_pos_l()
    self._update_phase_state()
    moving = torch.linalg.norm(command, dim=1) > self.cfg.reference_command_threshold
    left_swing = (self.swing_idx == 0) & moving
    right_swing = (self.swing_idx == 1) & moving
    swing_mask = torch.stack((left_swing, right_swing), dim=1)
    foot_contact = self._current_foot_contact()
    first_contact = foot_contact & ~self.prev_foot_contact
    landing_mask = self.prev_swing_foot_mask & first_contact
    landed = landing_mask.any(dim=1)
    if torch.any(landed):
      landed_foot_idx = torch.argmax(landing_mask.to(torch.long), dim=1)
      env_ids = torch.arange(self.num_envs, device=self.device)
      landing_target_xy_l = self.step_target_delta_xy[env_ids, landed_foot_idx, :]
      landing_actual_xy = self._current_foot_pos_w()[env_ids, landed_foot_idx, :2]
      landing_target_l = torch.zeros(self.num_envs, 3, device=self.device)
      landing_target_l[:, :2] = landing_target_xy_l
      landing_target_w = self.stance_foot_pos_0 + quat_apply(
        self.stance_foot_ori_quat_0,
        landing_target_l,
      )
      landing_target_xy = landing_target_w[:, :2]
      self.last_landing_actual_xy = torch.where(
        landed.unsqueeze(1),
        landing_actual_xy,
        self.last_landing_actual_xy,
      )
      self.last_landing_target_xy = torch.where(
        landed.unsqueeze(1),
        landing_target_xy,
        self.last_landing_target_xy,
      )
      self.last_landing_error_xy = torch.where(
        landed.unsqueeze(1),
        landing_actual_xy - landing_target_xy,
        self.last_landing_error_xy,
      )
      self.last_landing_valid = torch.where(
        landed,
        torch.ones_like(self.last_landing_valid),
        self.last_landing_valid,
      )
    self._update_stance_initial_pose()
    self._update_stance_state()
    foot_pos_l = self._current_foot_pos_l()
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
    delta_yaw = command[:, 2] * self.cur_swing_time
    cos_yaw = torch.cos(delta_yaw)
    sin_yaw = torch.sin(delta_yaw)
    target_delta_xy = torch.stack(
      (
        cos_yaw * target_delta_xy[:, 0] - sin_yaw * target_delta_xy[:, 1],
        sin_yaw * target_delta_xy[:, 0] + cos_yaw * target_delta_xy[:, 1],
      ),
      dim=1,
    )
    target_delta_xy_raw = target_delta_xy.clone()
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
    self.step_target_delta_xy = torch.where(
      swing_mask.unsqueeze(-1),
      target_delta_xy.unsqueeze(1),
      self.step_target_delta_xy,
    )

    horizontal_control = torch.tensor(
      (0.0, 0.0, 1.0, 1.0, 1.0),
      device=self.device,
      dtype=foot_pos_l.dtype,
    ).unsqueeze(0).expand(self.num_envs, -1)
    horizontal = torch.zeros_like(swing_phase)
    horizontal_dot = torch.zeros_like(swing_phase)
    for foot_idx in range(len(self.foot_body_names)):
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
    for foot_idx in range(len(self.foot_body_names)):
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
      self.num_envs, len(self.foot_body_names), 4
    )
    ref_pos_w = self.stance_foot_pos_0.unsqueeze(1) + quat_apply(
      stance_quat.reshape(-1, 4),
      ref_pos_l.reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)
    root_quat = self.robot.data.root_link_quat_w.unsqueeze(1).expand_as(stance_quat)
    root_pos = self.robot.data.root_link_pos_w.unsqueeze(1)
    ref_pos_b = quat_apply_inverse(
      root_quat.reshape(-1, 4),
      (ref_pos_w - root_pos).reshape(-1, 3),
    ).reshape(self.num_envs, len(self.foot_body_names), 3)
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
    x_clipped = (
      torch.abs(target_delta_xy_raw[:, 0] - target_delta_xy[:, 0]) > 1e-6
    ).float()
    self.metrics["step_ref_x"] = target_delta_xy[:, 0]
    self.metrics["step_ref_y"] = target_delta_xy[:, 1]
    self.metrics["landing_actual_x"] = self.last_landing_actual_xy[:, 0].clone()
    self.metrics["landing_actual_y"] = self.last_landing_actual_xy[:, 1].clone()
    self.metrics["landing_target_x"] = self.last_landing_target_xy[:, 0].clone()
    self.metrics["landing_target_y"] = self.last_landing_target_xy[:, 1].clone()
    self.metrics["landing_error_x"] = self.last_landing_error_xy[:, 0].clone()
    self.metrics["landing_error_y"] = self.last_landing_error_xy[:, 1].clone()
    self.metrics["landing_valid"] = self.last_landing_valid.clone()
    self.metrics["step_x_clipped"] = x_clipped * swing_mask.any(dim=1).float()
    pelvis_rpy_ref, pelvis_rpy_rate_ref = self._pelvis_reference(command)
    ref_swing_foot_rpy = torch.zeros_like(pelvis_rpy_ref)
    ref_swing_foot_rpy[:, 2] = pelvis_rpy_ref[:, 2]
    ref_swing_foot_rpy_rate = torch.zeros_like(ref_swing_foot_rpy)
    ref_swing_foot_rpy_rate[:, 2] = pelvis_rpy_rate_ref[:, 2]
    ref_upper_body_joint_pos, ref_upper_body_joint_vel = self._upper_body_reference(command)
    self._update_clf_state(
      command,
      self.phase,
      left_swing,
      foot_pos_b,
      ref_swing_foot_pos_l,
      ref_swing_foot_vel_l,
      ref_swing_foot_rpy,
      ref_swing_foot_rpy_rate,
      ref_upper_body_joint_pos,
      ref_upper_body_joint_vel,
    )
    self.prev_foot_rpy = self.foot_rpy.clone()
    self.prev_pelvis_rpy = self.pelvis_rpy.clone()
    self.swing_foot_phase = swing_phase
    self.swing_foot_mask = swing_mask
    self.prev_swing_foot_mask = swing_mask
    self.prev_foot_contact = foot_contact


@dataclass(kw_only=True)
class HLIPReferenceCommandCfg(CommandTermCfg):
  entity_name: str
  velocity_command_name: str = "twist"
  contact_sensor_name: str = "feet_ground_contact"
  reference_period: float = 0.7
  reference_command_threshold: float = 0.1
  foot_body_names: tuple[str, ...]
  swing_clearance: float = 0.12
  swing_step_x_min: float = -0.20
  swing_step_x_max: float = 0.35
  hlip_gravity: float = 9.81
  hlip_com_height: float = 0.61
  hlip_double_support_time: float = 0.1
  hlip_step_width: float = 0.26
  upper_body_joint_names: tuple[str, ...] = (
    "lumbar_yaw_.*",
    "left_shoulder_pitch_.*",
    "right_shoulder_pitch_.*",
    "left_shoulder_roll_.*",
    "right_shoulder_roll_.*",
    "left_shoulder_yaw_.*",
    "right_shoulder_yaw_.*",
    "left_elbow_pitch_.*",
    "right_elbow_pitch_.*",
  )
  waist_yaw_ref: float = 0.0
  shoulder_ref: tuple[float, float, float] = (0.16, 0.0, 0.0)
  elbow_ref: float = 0.1
  q_weights: tuple[float, ...] = HLIP_CLF_Q_WEIGHTS
  r_weights: tuple[float, ...] = HLIP_CLF_R_WEIGHTS
  yaw_idx: tuple[int, ...] = (5, 11)

  def build(self, env) -> HLIPReferenceCommand:
    return HLIPReferenceCommand(self, env)
