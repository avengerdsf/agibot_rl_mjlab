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
    ),
    stiffness=150.0,
    damping = 4.0,
    effort_limit=50.0,
    armature=0.01,
)

X1_ACTUATOR_FIXED_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "lumbar_roll_joint",
        "lumbar_pitch_joint",
    ),
    stiffness=700.0,
    damping=0.6,
    effort_limit=50.0,
    armature=0.01,
)

X1_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_elbow_pitch_joint",
    ),
    stiffness=40.0,
    damping=2.0,
    effort_limit=35.0,
    armature=0.01,
)

X1_ACTUATOR_FIXED_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_yaw_joint",
        ".*_elbow_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=30.0,
    damping=0.1,
    effort_limit=35.0,
    armature=0.01,
)

X1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_pitch_joint",
    ),
    stiffness=150.0,
    damping=2.0,
    effort_limit=120.0,
    armature=0.01,
)

X1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_roll_joint",
    ),
    stiffness=100.0,
    damping=2.0,
    effort_limit=120.0,
    armature=0.01,
)

X1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_yaw_joint",
    ),
    stiffness=150.0,
    damping=2.0,
    effort_limit=120.0,
    armature=0.01,
)

X1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_knee_pitch_joint",
    ),
    stiffness=150.0,
    damping=2.0,
    effort_limit=120.0,
    armature=0.01,
)

X1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_pitch_joint",
    ),
    stiffness=40.0,
    damping=2.0,
    effort_limit=80.0,
    armature=0.01,
)

X1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_roll_joint",
    ),
    stiffness=40.0,
    damping=2.0,
    effort_limit=80.0,
    armature=0.01,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.621),
  joint_pos={
    "lumbar_yaw_.*": 0.0,
    "lumbar_pitch_.*": 0.0,
    "lumbar_roll_.*": 0.0,
    "left_shoulder_pitch_.*": 0.15,
    "right_shoulder_pitch_.*": 0.15,
    "left_shoulder_roll_.*": -0.18,
    "right_shoulder_roll_.*": -0.18,
    ".*_shoulder_yaw_.*": 0.0,
    ".*_elbow_pitch_.*": 0.3,
    ".*_elbow_yaw_.*": 0.0,
    ".*_wrist_pitch_.*": 0.0,
    ".*_wrist_roll_.*": 0.0,
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
    geom_names_expr=(".*_collision",),
    condim={
        r"^(left|right)_foot_[1-7]_collision$": 6,
        r".*_collision$": 1,
    },
    priority={
        r"^(left|right)_foot_[1-7]_collision$": 1,
    },
    friction={
        r"^(left|right)_foot_[1-7]_collision$": (0.8,),
    },
)


X1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X1_ACTUATOR_WAIST,
    X1_ACTUATOR_FIXED_WAIST,
    X1_ACTUATOR_SHOULDER_PITCH,
    X1_ACTUATOR_FIXED_ARM,
    X1_ACTUATOR_HIP_PITCH,
    X1_ACTUATOR_HIP_ROLL,
    X1_ACTUATOR_HIP_YAW,
    X1_ACTUATOR_KNEE,
    X1_ACTUATOR_ANKLE_PITCH,
    X1_ACTUATOR_ANKLE_ROLL
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


X1_ACTION_SCALE: dict[str, float] = {
  r"lumbar_.*_joint": 0.125,
  r".*_shoulder_.*_joint": 0.21875,
  r".*_elbow_pitch_joint": 0.21875,
  r".*_hip_.*_joint": 0.25,
  r".*_knee_pitch_joint": 0.25,
  r".*_ankle_.*_joint": 0.25,
}
