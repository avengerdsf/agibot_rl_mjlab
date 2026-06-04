"""AgiBot X1 constants and asset loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from agibot_rl import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


def _default_x1_xml() -> Path:
  return SRC_PATH / "assets" / "robots" / "agibot_x1" / "xmls" / "x1.xml"


def resolve_x1_xml() -> Path:
  xml_from_env = os.environ.get("AGIBOT_X1_XML")
  if xml_from_env:
    xml_path = Path(xml_from_env).expanduser().resolve()
  else:
    xml_path = _default_x1_xml()
  if not xml_path.exists():
    raise FileNotFoundError(
      "AgiBot X1 MJCF XML not found. "
      "Put your manually imported model at "
      f"'{_default_x1_xml()}' or set AGIBOT_X1_XML to your XML path."
    )
  return xml_path


def get_assets(xml_path: Path, meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  mesh_root = (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent
  update_assets(assets, mesh_root, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  xml_path = resolve_x1_xml()
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = get_assets(xml_path, spec.meshdir)
  return spec


X1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "lumbar_yaw_joint",
    "lumbar_pitch_joint",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
)
X1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=35.0,
  armature=0.01,
)
X1_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_pitch_joint",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=120.0,
  armature=0.01,
)
X1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=80.0,
  armature=0.01,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.7),
  joint_pos={
    "lumbar_yaw_.*": 0.0,
    "lumbar_pitch_.*": 0.0,
    "left_shoulder_pitch_.*": 0.15,
    "right_shoulder_pitch_.*": 0.15,
    "left_shoulder_roll_.*": -0.18,
    "right_shoulder_roll_.*": -0.18,
    ".*_elbow_pitch_.*": 0.3,
    "left_hip_pitch_.*": 0.4,
    "right_hip_pitch_.*": -0.4,
    "left_hip_roll_.*": 0.05,
    "right_hip_roll_.*": -0.05,
    "left_hip_yaw_.*": -0.31,
    "right_hip_yaw_.*": 0.31,
    ".*_knee_pitch_.*": 0.49,
    ".*_ankle_pitch_.*": -0.21,
    ".*_ankle_roll_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim=3,
  priority=1,
  friction=(0.8,),
)


X1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X1_ACTUATOR_WAIST,
    X1_ACTUATOR_ARM,
    X1_ACTUATOR_LEG,
    X1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_x1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X1_ARTICULATION,
  )


X1_ACTION_SCALE: dict[str, float] = {}
for actuator in X1_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort_limit = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort_limit is not None
  for name in actuator.target_names_expr:
    X1_ACTION_SCALE[name] = 0.25 * effort_limit / stiffness
