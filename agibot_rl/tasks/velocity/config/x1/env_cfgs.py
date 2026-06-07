"""AgiBot X1 velocity environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
import torch
from agibot_rl.assets.robots import X1_ACTION_SCALE, get_x1_robot_cfg
from agibot_rl.tasks.velocity.human_base_env_cfg import POLICY_JOINTS_NAMES, make_velocity_env_cfg
from agibot_rl.tasks.velocity import mdp
from agibot_rl.tasks.velocity.mdp.commands.base_command import UniformVelocityCommandCfg as X1VelocityCommandCfg
from agibot_rl.tasks.velocity.mdp.commands.gait_phase_command import GaitPhaseCommandCfg
from agibot_rl.tasks.velocity.mdp.commands.hlip_reference_command import HLIPReferenceCommandCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg


X1_FOOT_COLLISION_GEOMS = (
  "left_foot_1_collision",
  "left_foot_2_collision",
  "left_foot_3_collision",
  "left_foot_4_collision",
  "left_foot_5_collision",
  "left_foot_6_collision",
  "left_foot_7_collision",
  "right_foot_1_collision",
  "right_foot_2_collision",
  "right_foot_3_collision",
  "right_foot_4_collision",
  "right_foot_5_collision",
  "right_foot_6_collision",
  "right_foot_7_collision",
)
X1_FOOT_SITES = ("left_foot", "right_foot")
X1_FOOT_BODIES = ("link_left_ankle_roll", "link_right_ankle_roll")

def stacked_term_obs(
    env,
    term,
    history_length: int,
    buffer_name: str,
) -> torch.Tensor:
  """对单个 ObservationTermCfg 做历史堆叠，保留默认表格的 term 展开显示。"""

  obs_now = term.func(env, **(term.params or {}))

  if term.scale is not None:
    obs_now = obs_now * term.scale

  obs_now = obs_now.reshape(obs_now.shape[0], -1)

  step_name = f"{buffer_name}_step"
  step_now = env.episode_length_buf.clone()

  if not hasattr(env, buffer_name) or not hasattr(env, step_name):
    buffer = obs_now.unsqueeze(1).repeat(1, history_length, 1)
    setattr(env, buffer_name, buffer)
    setattr(env, step_name, step_now)
    return buffer.reshape(obs_now.shape[0], -1)

  buffer = getattr(env, buffer_name)
  last_step = getattr(env, step_name)

  shape_changed = (
      buffer.shape[0] != obs_now.shape[0]
      or buffer.shape[1] != history_length
      or buffer.shape[2] != obs_now.shape[1]
      or last_step.shape[0] != step_now.shape[0]
  )

  if shape_changed:
    buffer = obs_now.unsqueeze(1).repeat(1, history_length, 1)
    setattr(env, buffer_name, buffer)
    setattr(env, step_name, step_now)
    return buffer.reshape(obs_now.shape[0], -1)

  update_mask = step_now != last_step
  reset_mask = step_now == 0

  if torch.any(update_mask):
    updated = torch.roll(buffer[update_mask], shifts=1, dims=1)
    updated[:, 0, :] = obs_now[update_mask]
    buffer[update_mask] = updated

  if torch.any(reset_mask):
    buffer[reset_mask] = obs_now[reset_mask].unsqueeze(1).repeat(1, history_length, 1)

  setattr(env, buffer_name, buffer)
  setattr(env, step_name, step_now)

  return buffer.reshape(obs_now.shape[0], -1)

def stack_observation_terms(
    terms: dict,
    history_length: int,
    group_name: str,
) -> dict:
  """把一组 ObservationTermCfg 改成逐项历史堆叠。"""

  return {
    name: ObservationTermCfg(
      func=stacked_term_obs,
      params={
        "term": term,
        "history_length": history_length,
        "buffer_name": f"_{group_name}_{name}_history",
      },
    )
    for name, term in terms.items()
  }

def agibot_x1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # 在 env_cfgs.py 的 agibot_x1_flat_env_cfg() 中添加
  cfg.sim.mujoco.iterations = 100     # MuJoCo 默认值
  cfg.sim.mujoco.ls_iterations = 50   # MuJoCo 默认值

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_x1_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "x1-body"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(link_left_ankle_roll|link_right_ankle_roll)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  feet_slip_cfg = ContactSensorCfg(
    name="feet_slip_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left_foot_[1-7]_collision|right_foot_[1-7]_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="x1-body", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="x1-body", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  torso_ground_cfg = ContactSensorCfg(
    name="torso_ground_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(x1-body|link_lumbar_yaw|link_lumbar_roll|link_lumbar_pitch)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (
    cfg.scene.sensors or ()
  ) + (feet_ground_cfg, feet_slip_cfg, self_collision_cfg, torso_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # joint_pos_action.scale = 0.5
  joint_pos_action.scale = X1_ACTION_SCALE

  cfg.viewer.body_name = "link_lumbar_pitch"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, X1VelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0
  cfg.commands["gait_phase"] = GaitPhaseCommandCfg(
    resampling_time_range=(1e9, 1e9),
    period=0.7,
  )
  cfg.observations["actor"].terms["phase"].func = mdp.generated_commands
  cfg.observations["actor"].terms["phase"].params = {"command_name": "gait_phase"}
  cfg.observations["critic"].terms["phase"].func = mdp.generated_commands
  cfg.observations["critic"].terms["phase"].params = {"command_name": "gait_phase"}
  cfg.commands["hlip_ref"] = HLIPReferenceCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    velocity_command_name="twist",
    reference_period=0.7,
    reference_command_threshold=0.1,
    foot_body_names=X1_FOOT_BODIES,
    swing_clearance=0.12,
    swing_step_x_min=-0.25,
    swing_step_x_max=0.50,
    hlip_com_height=0.61,
    hlip_double_support_time=0.1,
    hlip_step_width=0.26,
  )

  cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    geom_names=X1_FOOT_COLLISION_GEOMS,
    preserve_order=True,
  )
  cfg.events["foot_friction"].params["ranges"] = (0.8,1.5)
  cfg.curriculum.pop("terrain_levels", None)

  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  def foot_site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      site_names=X1_FOOT_SITES,
      preserve_order=True,
    )

  def foot_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=X1_FOOT_BODIES,
      preserve_order=True,
    )

  foot_geom_cfg = SceneEntityCfg(
    "robot",
    geom_names=X1_FOOT_COLLISION_GEOMS,
    preserve_order=True,
  )

  cfg.observations["critic"].terms.pop("foot_height", None)
  cfg.observations["critic"].terms.pop("foot_air_time", None)
  # cfg.observations["critic"].terms["foot_height"].params["asset_cfg"] = foot_site_cfg()
  cfg.observations["critic"].terms.pop("gait_reference_joint_pos_error", None)
  cfg.observations["critic"].terms["hlip_ref_traj"] = ObservationTermCfg(
    func=mdp.hlip_ref_traj,
    params={"command_name": "hlip_ref", "swing_z_scale": 25.0},
  )
  cfg.observations["critic"].terms["hlip_act_traj"] = ObservationTermCfg(
    func=mdp.hlip_act_traj,
    params={"command_name": "hlip_ref", "swing_z_scale": 25.0},
  )
  cfg.observations["critic"].terms["hlip_ref_traj_vel"] = ObservationTermCfg(
    func=mdp.hlip_ref_traj_vel,
    params={"command_name": "hlip_ref"},
    clip=(-20.0, 20.0),
  )
  cfg.observations["critic"].terms["hlip_act_traj_vel"] = ObservationTermCfg(
    func=mdp.hlip_act_traj_vel,
    params={"command_name": "hlip_ref"},
    clip=(-20.0, 20.0),
  )
  cfg.observations["critic"].terms["foot_vel"] = ObservationTermCfg(
    func=mdp.foot_vel,
    params={"asset_cfg": foot_body_cfg()},
  )
  obs_history_length = 8

  cfg.observations["actor"].terms = stack_observation_terms(
    terms=cfg.observations["actor"].terms,
    history_length=obs_history_length,
    group_name="actor",
  )

  cfg.observations["critic"].terms = stack_observation_terms(
    terms=cfg.observations["critic"].terms,
    history_length=obs_history_length,
    group_name="critic",
  )

  cfg.observations["actor"].history_length = 1
  cfg.observations["critic"].history_length = 1

  cfg.observations["actor"].enable_corruption = False
  cfg.observations["critic"].enable_corruption = False


  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.16)
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
  cfg.rewards["track_vel_hard"].params["sigma_v"] = 0.30
  cfg.rewards["track_vel_hard"].weight = 0.80
  # cfg.rewards["track_linear_velocity"] = None
  # cfg.rewards["track_angular_velocity"] = None
  # cfg.rewards["track_vel_hard"]  = None




  cfg.rewards["base_acc"] = RewardTermCfg(
    func=mdp.base_acc_l2,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "log_prefix": "Metrics/base_acc",
    },
  )
  # cfg.rewards["base_heading_tracking"] = RewardTermCfg(
  #   func=mdp.base_heading_tracking,
  #   weight=0.5,
  #   params={
  #     "command_name": "twist",
  #     "sigma": 0.5,
  #     "command_threshold": 0.05,
  #     "reward_scale": 2.0,
  #   },
  # )
  cfg.rewards["yaw_rate_zero_command"] = RewardTermCfg(
    func=mdp.yaw_rate_zero_command_penalty,
    weight=-1.0,
    params={"command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["foot_gait"] = None
  cfg.rewards["foot_clearance"] = None
  # cfg.rewards["feet_swing_height"] = RewardTermCfg(
  #   func=mdp.feet_swing_height,
  #   weight=-1.0,
  #   params={
  #     "sensor_name": feet_ground_cfg.name,
  #     "target_height": 0.10,
  #     "command_name": "twist",
  #     "command_threshold": 0.1,
  #     "asset_cfg": foot_site_cfg(),
  #   },
  # )
  cfg.rewards["feet_swing_height"] = None
  cfg.rewards["swing_foot_trajectory"] = RewardTermCfg(
    func=mdp.swing_foot_trajectory,
    weight=0.5,
    params={
      "command_name": "hlip_ref",
      "std": 0.08,
      "command_threshold": 0.1,
      "asset_cfg": foot_body_cfg(),
    },
  )
  cfg.rewards["hlip_clf_reward"] = RewardTermCfg(
    func=mdp.clf_reward,
    weight=10.0,
    params={
      "command_name": "hlip_ref",
      "max_eta_err": 0.25,
    },
  )
  cfg.rewards["hlip_clf_decreasing_condition"] = RewardTermCfg(
    func=mdp.clf_decreasing_condition,
    weight=-2.0,
    params={
      "command_name": "hlip_ref",
      "alpha": 0.5,
      "eta_max": 0.2,
      "eta_dot_max": 0.3,
    },
  )
  cfg.rewards["hlip_holonomic_constraint"] = RewardTermCfg(
    func=mdp.holonomic_constraint,
    weight=0.4,
    params={
      "command_name": "hlip_ref",
      "sigma_pose": math.sqrt(5.0 * 0.01),
    },
  )
  cfg.rewards["hlip_holonomic_constraint_vel"] = RewardTermCfg(
    func=mdp.holonomic_constraint_vel,
    weight=0.2,
    params={
      "command_name": "hlip_ref",
      "sigma_vel": math.sqrt(0.1),
    },
  )
  cfg.rewards["foot_slip"].func = mdp.feet_geom_slip
  cfg.rewards["foot_slip"].params["sensor_name"] = feet_slip_cfg.name
  cfg.rewards["foot_slip"].params["asset_cfg"] = foot_geom_cfg
  cfg.rewards["foot_slip"].params["num_feet"] = 2
  cfg.rewards["foot_slip"].params["period"] = 0.7
  cfg.rewards["foot_slip"].params["offset"] = [0.0, 0.5]
  cfg.rewards["foot_slip"].params["threshold"] = 0.56
  cfg.rewards["foot_slip"].params["min_contact_fraction"] = 0.5
  cfg.rewards["foot_slip"].weight = -0.05

  cfg.rewards["action_rate_l2"].weight = -0.002

  cfg.rewards["stance_foot_contact"] = RewardTermCfg(
    func=mdp.stance_foot_contact_count_penalty,
    weight=-2.0,
    params={
      "sensor_name": feet_slip_cfg.name,
      "period": 0.7,
      "offset": [0.0, 0.5],
      "threshold": 0.56,
      "command_name": "twist",
      "command_threshold": 0.1,
      "num_feet": 2,
    },
  )
  cfg.rewards["feet_contact_number"] = None
  cfg.rewards["gait_reference_joint_pos"] = None

  cfg.rewards["body_height_l2"] = RewardTermCfg(
    func=mdp.body_height_l2,
    weight=-0.2,
    params={
      "target_height": 0.61,
      "std": 0.08,
      "asset_cfg": SceneEntityCfg("robot", body_names=("x1-body",)),
    },
  )

  cfg.metrics["body_height"] = MetricsTermCfg(
    func=mdp.body_height,
    params={
      "target_height": 0.61,
      "asset_cfg": SceneEntityCfg("robot", body_names=("x1-body",))
      },
  )
  # cfg.metrics["yaw_roll_actuator_force_ratio"] = MetricsTermCfg(
  #   func=mdp.actuator_force_ratio,
  #   params={
  #     "sensor_names": [
  #       "robot/jointeffort_lumbar_yaw",
  #       "robot/jointeffort_left_hip_yaw",
  #       "robot/jointeffort_right_hip_yaw",
  #       "robot/jointeffort_left_hip_roll",
  #       "robot/jointeffort_right_hip_roll",
  #       "robot/jointeffort_left_ankle_roll",
  #       "robot/jointeffort_right_ankle_roll",
  #     ],
  #     "effort_limits": [50.0, 120.0, 120.0, 120.0, 120.0, 80.0, 80.0],
  #     "log_prefix": "Metrics/yaw_roll_actuator_force_ratio",
  #     "log_per_sensor": True,
  #   },
  # )
  # cfg.metrics["policy_actuator_force_ratio"] = MetricsTermCfg(
  #   func=mdp.actuator_force_ratio,
  #   params={
  #     "sensor_names": [
  #       "robot/jointeffort_lumbar_yaw",
  #       "robot/jointeffort_left_shoulder_pitch",
  #       "robot/jointeffort_right_shoulder_pitch",
  #       "robot/jointeffort_left_shoulder_roll",
  #       "robot/jointeffort_right_shoulder_roll",
  #       "robot/jointeffort_left_elbow_pitch",
  #       "robot/jointeffort_right_elbow_pitch",
  #       "robot/jointeffort_left_hip_pitch",
  #       "robot/jointeffort_right_hip_pitch",
  #       "robot/jointeffort_left_hip_roll",
  #       "robot/jointeffort_right_hip_roll",
  #       "robot/jointeffort_left_hip_yaw",
  #       "robot/jointeffort_right_hip_yaw",
  #       "robot/jointeffort_left_knee_pitch",
  #       "robot/jointeffort_right_knee_pitch",
  #       "robot/jointeffort_left_ankle_pitch",
  #       "robot/jointeffort_right_ankle_pitch",
  #       "robot/jointeffort_left_ankle_roll",
  #       "robot/jointeffort_right_ankle_roll",
  #     ],
  #     "effort_limits": [
  #       50.0,
  #       35.0,
  #       35.0,
  #       35.0,
  #       35.0,
  #       35.0,
  #       35.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       120.0,
  #       80.0,
  #       80.0,
  #       80.0,
  #       80.0,
  #     ],
  #     "log_prefix": "Metrics/policy_actuator_force_ratio",
  #     "log_per_sensor": True,
  #   },
  # )

  cfg.events["base_com"].params["asset_cfg"].body_names = ("x1-body",)

  # cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
  #   "robot",
  #   joint_names=(
  #     "lumbar_yaw_.*",
  #     "left_shoulder_pitch_.*",
  #     "right_shoulder_pitch_.*",
  #     "left_shoulder_roll_.*",
  #     "right_shoulder_roll_.*",
  #     ".*_elbow_pitch_.*",
  #     "left_hip_roll_.*",
  #     "right_hip_roll_.*",
  #     "left_hip_yaw_.*",
  #     "right_hip_yaw_.*",
  #     ".*_ankle_roll_.*",
  #   ),
  # )
  # cfg.rewards["pose"].params["std_standing"] = {".*": 0.1}
  # cfg.rewards["pose"].params["std_walking"] = {
  #   r".*_hip_roll_.*$": 0.15,
  #   r".*_hip_yaw_.*$": 0.15,
  #   r".*_ankle_roll_.*$": 0.15,
  #   r"lumbar_.*": 0.15,
  #   r".*_shoulder_.*": 0.2,
  #   r".*_elbow_.*": 0.1,
  # }
  # cfg.rewards["pose"].params["std_running"] = {
  #   r".*_hip_roll_.*$": 0.25,
  #   r".*_hip_yaw_.*$": 0.25,
  #   r".*_ankle_roll_.*$": 0.15,
  #   r"lumbar_.*": 0.15,
  #   r".*_shoulder_.*": 0.25,
  #   r".*_elbow_.*": 0.2,
  # }
  cfg.rewards["pose"] = None
  cfg.rewards["x1_joint_default_pos"] = None
  cfg.rewards["x1_joint_vel_l2"] = None
  fixed_joint_names = (
    "lumbar_roll_.*",
    "lumbar_pitch_.*",
    ".*_shoulder_yaw_.*",
    ".*_elbow_yaw_.*",
    ".*_wrist_pitch_.*",
    ".*_wrist_roll_.*",
  )
  cfg.rewards["x1_fixed_joint_pos_l2"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-0.5,
    params={
      "command_name": None,
      "asset_cfg": SceneEntityCfg("robot", joint_names=fixed_joint_names),
    },
  )
  cfg.rewards["x1_fixed_joint_vel_l2"] = RewardTermCfg(
    func=mdp.x1_joint_vel_l2,
    weight=-0.02,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=fixed_joint_names),
      "log_prefix": "Metrics/x1_fixed_joint_vel_l2",
    },
  )
  cfg.rewards["x1_lumbar_yaw_limit"] = RewardTermCfg(
    func=mdp.joint_pos_limit,
    weight=-1.0,
    params={
      "limit": 0.45,
      "asset_cfg": SceneEntityCfg("robot", joint_names=("lumbar_yaw_.*",)),
      "log_prefix": "Metrics/x1_lumbar_yaw_limit",
    },
  )

  cfg.rewards["body_orientation_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("x1-body",)
  )
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("x1-body",)
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # cfg.terminations["torso_ground_contact"] = TerminationTermCfg(
  #   func=mdp.illegal_contact,
  #   params={"sensor_name": torso_ground_cfg.name, "force_threshold": 10.0},
  # )

  cfg.rewards["angular_momentum"].weight = -1e-3


  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None


  

  if play:
    # cfg.episode_length_s = int(1e9)
    # cfg.observations["actor"].enable_corruption = False
    # cfg.events.pop("push_robot", None)
    # cfg.curriculum = {}
    # cfg.events["randomize_terrain"] = EventTermCfg(
    #   func=envs_mdp.randomize_terrain,
    #   mode="reset",
    #   params={},
    # )
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, X1VelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0,0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)

  return cfg
