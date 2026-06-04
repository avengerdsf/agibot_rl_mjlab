from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def gait_reference_joint_pos_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  if not hasattr(command_term, "ref_joint_pos"):
    raise RuntimeError(
      "gait_reference_joint_pos_error requires a command term with ref_joint_pos."
    )

  asset = command_term.robot
  joint_ids = command_term.ref_joint_ids_tensor
  return asset.data.joint_pos[:, joint_ids] - command_term.ref_joint_pos


def hlip_ref_traj(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.y_out


def hlip_act_traj(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.y_act


def hlip_ref_traj_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.dy_out


def hlip_act_traj_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.dy_act


def hlip_clf_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return torch.stack((command_term.v, command_term.vdot), dim=1)


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
