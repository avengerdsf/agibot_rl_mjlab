"""Script to train RL agent with RSL-RL."""

import logging
import os
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_MODE"] = "online"
print(f"[INFO] W&B API Key injected: {os.environ['WANDB_API_KEY'][:5]}...")
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import mjlab
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent
  resume_path: Path | None = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    runner.load(str(resume_path))

  # if rank == 0:
  #   dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  #   dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir or str(log_dir / "torchrunx")

    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


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
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  launch_training(chosen_task, args)


if __name__ == "__main__":
  main()
