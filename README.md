# AgiBot RL Mjlab

这是基于 `/home/chen/code/my_robot/robot_rl_mjlab` 搭起来的轻量 `mjlab` 强化学习框架，目录组织、训练入口和配置风格保持一致，但机器人模型不随工程拷贝。

## 当前提供

- 训练入口：`scripts/train.py`
- 回放入口：`scripts/play.py`
- 任务注册：`AgiBot-X1-Flat`
- 机器人配置：`agibot_rl/assets/robots/agibot_x1/x1_constants.py`
- 环境配置：`agibot_rl/tasks/velocity/config/x1/env_cfgs.py`
- PPO 配置：`agibot_rl/tasks/velocity/config/x1/rl_cfg.py`

## 你需要手动导入的模型

把 X1 的 MJCF XML 放到：

```bash
agibot_rl/assets/robots/agibot_x1/xmls/x1.xml
```

或者设置环境变量：

```bash
export AGIBOT_X1_XML=/abs/path/to/x1.xml
```

要求：

- XML 能被 MuJoCo 正常编译
- mesh 相对路径和 `meshdir` 配置正确
- 机器人至少包含这些关节名模式：
  `lumber_yaw`、`lumber_pitch`、`left/right_hip_*`、`left/right_knee_pitch`、`left/right_ankle_*`
- 建议保留脚部 body 名：
  `left_ankle_roll_link`、`right_ankle_roll_link`

## 使用方式

安装：

```bash
pip install -e .
```

查看任务：

```bash
python scripts/list_envs.py
```

训练：

```bash
python scripts/train.py AgiBot-X1-Flat --env.scene.num-envs=4096
```

回放：

```bash
python scripts/play.py AgiBot-X1-Flat --checkpoint_file=logs/rsl_rl/agibot_x1_velocity/<run>/model_<iter>.pt
```

## 说明

- 这个版本先按平地速度跟踪任务搭好了主干。
- 为了降低你后续接入自己模型的成本，脚高相关观测和奖励先关掉了，只保留了接触、速度跟踪、姿态和 PPO 训练主链路。
- 如果你导入后的 XML 命名和这里假设的不一致，优先改 `x1_constants.py` 和 `env_cfgs.py` 里的 body/joint 正则。
