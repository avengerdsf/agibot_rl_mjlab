from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)

def bad_orientation_x1(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "link_lumbar_pitch",
):
    """Terminate when the specified body's orientation exceeds the limit angle."""
    asset: Entity = env.scene[asset_cfg.name]

    # 不要 resolve 传进来的 asset_cfg，避免反复改同一个配置对象
    body_cfg = SceneEntityCfg(asset_cfg.name, body_names=(body_name,))
    body_cfg.resolve(env.scene)
    body_id = body_cfg.body_ids[0]

    # world -> body
    # xmat: (nworld, nbody, 3, 3), 是 body->world 旋转矩阵 R_wb
    # 所以 R_bw = R_wb^T
    R_wb = asset.data.data.xmat[:, body_id, :, :]               # (nworld, 3, 3)
    gravity_w = asset.data.gravity_vec_w                        # (nworld, 3), 一般是 [0,0,-1]

    gravity_b = torch.bmm(R_wb.transpose(1, 2), gravity_w.unsqueeze(-1)).squeeze(-1)  # (nworld, 3)

    # 和原版保持一致：upright 时 gravity_b ≈ [0,0,-1]，所以看 -gravity_b[:,2]
    cos_angle = torch.clamp(-gravity_b[:, 2], -1.0, 1.0)
    angle = torch.acos(cos_angle)
    return angle > limit_angle


def bad_orientation_body(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when body orientation exceeds the limit angle."""

  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    batch_size = body_quat_w.shape[0]
    num_bodies = body_quat_w.shape[1]

    gravity_w = asset.data.gravity_vec_w

    if gravity_w.dim() == 1:
      gravity_w = gravity_w.view(1, 1, 3).expand(batch_size, num_bodies, 3)
    elif gravity_w.dim() == 2:
      gravity_w = gravity_w.view(batch_size, 1, 3).expand(batch_size, num_bodies, 3)
    else:
      gravity_w = gravity_w.expand(batch_size, num_bodies, 3)

    projected_gravity_b = quat_apply_inverse(
      body_quat_w.reshape(-1, 4),
      gravity_w.reshape(-1, 3),
    ).reshape(batch_size, num_bodies, 3)

    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :, :2]), dim=-1)
    orientation_l2 = torch.max(xy_squared, dim=1).values
  else:
    orientation_l2 = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

  threshold = math.sin(limit_angle) ** 2
  return orientation_l2 > threshold