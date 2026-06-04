"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import inspect
import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# from utils.pub_points import Mid360SocketEnvWrapper
# from utils.pub_imu import ImuSocketEnvWrapper




@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  # Real gamepad override (disabled by default).
  gamepad: bool = False
  gamepad_device: str = "/dev/input/js0"
  gamepad_type: Literal["xbox", "switch"] = "xbox"
  gamepad_deadzone: float = 0.05
  gamepad_scale_x: float = 1.0
  gamepad_scale_yaw: float = 1.0
  gamepad_command_term: str = "twist"


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.gamepad:
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)


  if cfg.no_terminations:
    env_cfg.terminations = {}

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  dummy_mode = cfg.agent in {"zero", "random"}
  render_mode = "rgb_array" if (cfg.video and not dummy_mode) else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video and not dummy_mode and cfg.checkpoint_file is not None:
    log_dir = Path(cfg.checkpoint_file).resolve().parent
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  # 模拟360点云消息
  has_mid360 = any(s.name == "mid360" for s in (env_cfg.scene.sensors or ()))
  # if has_mid360:
    # env = Mid360SocketEnvWrapper(
    #   env,
    #   sensor_name="mid360",
    #   host="127.0.0.1",
    #   port=8765,
    #   send_hz=15.0,
    # )
    # env = ImuSocketEnvWrapper(
    #   env,
    #   ang_vel_sensor_name="robot/imu_ang_vel",
    #   lin_acc_sensor_name="robot/imu_lin_acc",
    #   quat_sensor_name=None,  # 如果后面有四元数传感器再改
    #   host="127.0.0.1",
    #   port=8766,
    #   send_hz=200.0,
    # )
  action_shape = env.unwrapped.action_space.shape

  if cfg.agent == "zero":
    class PolicyZero:
      def __call__(self, obs):
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)
    policy = PolicyZero()
  elif cfg.agent == "random":
    class PolicyRandom:
      def __call__(self, obs):
        del obs
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
    policy = PolicyRandom()
  else:
    if cfg.checkpoint_file is None:
      raise ValueError("--checkpoint-file is required when agent=trained")
    resume_path = Path(cfg.checkpoint_file).resolve()
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    resolved_viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  else:
    resolved_viewer = cfg.viewer

  gamepad_reader = None
  if cfg.gamepad:
    from utils.gamepad import make_gamepad
    gamepad_reader = make_gamepad(
      device=cfg.gamepad_device,
      gamepad_type=cfg.gamepad_type,
      deadzone=cfg.gamepad_deadzone,
      scale_x=cfg.gamepad_scale_x,
      scale_yaw=cfg.gamepad_scale_yaw,
      autostart=True,
    )
    try:
      print("1"*100)
      vel_cmd = env.unwrapped.command_manager.get_term(cfg.gamepad_command_term)
      print("[gamepad] term type =", type(vel_cmd))
      print("[gamepad] term file =", inspect.getfile(type(vel_cmd)))
      print("[gamepad] dir has method =", "set_gamepad_source" in dir(vel_cmd))
      print("[gamepad] class has method =", hasattr(type(vel_cmd), "set_gamepad_source"))
      print("[gamepad] instance has method =", hasattr(vel_cmd, "set_gamepad_source"))
      print(inspect.getsource(type(vel_cmd)))

      vel_cmd.set_gamepad_source(gamepad_reader, get_env_idx=lambda: 0, enabled=True)
    except (AttributeError, KeyError) as e:
      print(f"[gamepad] Warning: could not attach to command term '{cfg.gamepad_command_term}': {e}")
      gamepad_reader.stop()
      gamepad_reader = None

    print("2"*100)
    

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    if gamepad_reader is not None:
      gamepad_reader.stop()

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import agibot_rl.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
