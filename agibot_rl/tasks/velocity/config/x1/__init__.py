from mjlab.tasks.registry import register_mjlab_task
from agibot_rl.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import agibot_x1_flat_env_cfg
from .rl_cfg import agibot_x1_ppo_runner_cfg


register_mjlab_task(
  task_id="AgiBot-X1-Flat",
  env_cfg=agibot_x1_flat_env_cfg(),
  play_env_cfg=agibot_x1_flat_env_cfg(play=True),
  rl_cfg=agibot_x1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
