from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import mujoco
import numpy as np


# 环境变量名称。
# 如果不想使用默认项目路径，可以通过 AGIBOT_X1_XML 指定 x1.xml 的绝对路径。
ENV_XML_VAR = "AGIBOT_X1_XML"


# X1 的默认初始关节角。
# 这里的数值应与训练配置中的 HOME_KEYFRAME 保持一致。
# 本脚本只做姿态坐标系验证，因此不需要控制器和动作输入。
HOME_JOINT_POS = {
    r"lumbar_yaw_.*": 0.0,
    r"lumbar_roll_.*": 0.0,
    r"left_shoulder_pitch_.*": 0.15,
    r"right_shoulder_pitch_.*": 0.15,
    r"left_shoulder_roll_.*": -0.18,
    r"right_shoulder_roll_.*": -0.18,
    r".*_elbow_pitch_.*": 0.3,
    r"left_hip_pitch_.*": 0.4,
    r"right_hip_pitch_.*": -0.4,
    r"left_hip_roll_.*": 0.05,
    r"right_hip_roll_.*": -0.05,
    r"left_hip_yaw_.*": -0.31,
    r"right_hip_yaw_.*": 0.31,
    r".*_knee_pitch_.*": 0.49,
    r".*_ankle_pitch_.*": -0.21,
    r".*_ankle_roll_.*": 0.0,
}


def resolve_xml_path(xml_arg: str | None) -> Path:
    """解析 AgiBot X1 的 MJCF XML 路径。

    优先级：
    1. 命令行参数 --xml
    2. 环境变量 AGIBOT_X1_XML
    3. agibot_rl.SRC_PATH 下的默认模型路径

    这样做可以同时兼容：
    1. 在项目环境中直接运行
    2. 手动指定某个测试 XML
    3. 用环境变量切换不同版本的 X1 模型
    """

    # 最高优先级：命令行显式传入的 XML 路径。
    if xml_arg:
        return Path(xml_arg).expanduser().resolve()

    # 第二优先级：环境变量指定的 XML 路径。
    xml_from_env = os.environ.get(ENV_XML_VAR)
    if xml_from_env:
        return Path(xml_from_env).expanduser().resolve()

    # 第三优先级：从 agibot_rl 包中读取 SRC_PATH。
    # 这是项目内部最稳定的寻址方式。
    try:
        from agibot_rl import SRC_PATH

        return (
            Path(SRC_PATH)
            / "assets"
            / "robots"
            / "agibot_x1"
            / "xmls"
            / "x1.xml"
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            "Cannot resolve X1 XML path. Use --xml or set AGIBOT_X1_XML."
        ) from exc


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 roll、pitch、yaw 转换为 MuJoCo 使用的四元数格式。

    MuJoCo 的自由关节 qpos 中，姿态四元数顺序为：

    [w, x, y, z]

    输入角度单位为 rad。
    """

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float64)


def get_joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """根据 MuJoCo joint id 获取关节名称。

    MuJoCo 在找不到名称时可能返回 None，这里统一转换为空字符串，
    方便后续正则匹配。
    """

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
        return ""
    return name


def set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: tuple[float, float, float],
    quat: np.ndarray,
) -> None:
    """设置 floating base 的位置和姿态。

    X1 的根关节是 freejoint，它在 qpos 中占 7 个位置：

    qpos[0:3]  根部位置
    qpos[3:7]  根部四元数 [w, x, y, z]

    本脚本通过直接写 qpos 的方式把机器人悬空放置，
    然后调用 mj_forward 计算所有 body 的世界坐标系。
    """

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Freejoint not found: {joint_name}")

    qpos_adr = model.jnt_qposadr[joint_id]

    data.qpos[qpos_adr : qpos_adr + 3] = np.array(pos, dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat


def set_home_joints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """设置机器人所有非 freejoint 关节到 HOME_JOINT_POS。

    这里按正则表达式匹配 joint name。
    没有匹配到的关节默认置 0。

    注意：
    这个函数只设置 qpos，不设置控制器目标。
    因为本脚本只用于静态姿态坐标系验证，不进行动力学仿真。
    """

    for joint_id in range(model.njnt):
        joint_name = get_joint_name(model, joint_id)
        joint_type = model.jnt_type[joint_id]

        # freejoint 由 set_freejoint_pose 单独设置。
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue

        qpos_adr = model.jnt_qposadr[joint_id]

        matched = False
        for pattern, value in HOME_JOINT_POS.items():
            if re.fullmatch(pattern, joint_name):
                data.qpos[qpos_adr] = float(value)
                matched = True
                break

        if not matched:
            data.qpos[qpos_adr] = 0.0


def body_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    """读取指定 body 的世界旋转矩阵。

    data.xmat[body_id] 是 MuJoCo 计算后的 body 世界旋转矩阵，
    展开存储为长度 9 的数组。

    reshape(3, 3) 后：
    第 1 列是 body 局部 x 轴在世界系下的方向
    第 2 列是 body 局部 y 轴在世界系下的方向
    第 3 列是 body 局部 z 轴在世界系下的方向
    """

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {body_name}")

    return data.xmat[body_id].reshape(3, 3).copy()


def body_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    """读取指定 body 的世界位置。"""

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {body_name}")

    return data.xpos[body_id].copy()


def orientation_l2_like(R_world_body: np.ndarray) -> float:
    """计算一个近似 body_orientation_l2 的姿态误差。

    常见的 upright penalty 会把世界重力方向投影到 body 坐标系，
    然后惩罚 body 坐标系下的 x、y 分量：

    orientation_l2 = g_body_x^2 + g_body_y^2

    当 body 的局部 z 轴竖直向上时：
    g_body = [0, 0, -1]
    orientation_l2 = 0

    当 body 的局部 z 轴水平时：
    g_body 的 x 或 y 分量接近 1
    orientation_l2 接近 1

    这个函数用于判断某个 body 是否适合作为姿态奖励参考体。
    """

    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity_body = R_world_body.T @ gravity_world

    return float(gravity_body[0] ** 2 + gravity_body[1] ** 2)


def upright_dot(R_world_body: np.ndarray) -> float:
    """计算 body 局部 z 轴和世界 z 轴的夹角余弦。

    返回值含义：
    1      body 的 z 轴竖直向上
    0      body 的 z 轴水平
    -1     body 的 z 轴竖直向下

    正常直立状态下，适合作为 base 姿态参考的 body 应接近 1。
    """

    body_z_world = R_world_body[:, 2]
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return float(body_z_world @ world_z)


def rotation_angle(R: np.ndarray) -> float:
    """根据相对旋转矩阵计算旋转角度。

    输入 R 是两个 body 坐标系之间的相对旋转矩阵。
    返回角度单位为 rad。
    """

    cos_angle = (np.trace(R) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    return math.acos(cos_angle)


def print_case(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    case_name: str,
    roll: float,
    pitch: float,
    yaw: float,
    root_z: float,
) -> None:
    """执行一个测试姿态，并打印两个 body 的姿态信息。

    每个 case 会：
    1. 重置 MuJoCo data
    2. 设置 floating base 的位置和姿态
    3. 设置所有关节到 HOME_JOINT_POS
    4. 调用 mj_forward 更新运动学
    5. 打印 x1-body 和 link_lumbar_pitch 的世界坐标轴
    6. 计算两者之间的相对旋转角度

    这个函数不调用 mj_step，所以没有动力学积分和接触影响。
    """

    quat = quat_from_rpy(roll, pitch, yaw)

    mujoco.mj_resetData(model, data)

    set_freejoint_pose(
        model,
        data,
        joint_name="floating_base",
        pos=(0.0, 0.0, root_z),
        quat=quat,
    )

    set_home_joints(model, data)

    # 只做前向运动学和传感器相关计算。
    # 不进行仿真步进，不会产生接触动力学影响。
    mujoco.mj_forward(model, data)

    bodies = ("x1-body", "link_lumbar_pitch")
    results: dict[str, np.ndarray] = {}

    print(f"\ncase: {case_name}")
    print(
        "root_rpy_deg:",
        f"roll={math.degrees(roll):.2f}",
        f"pitch={math.degrees(pitch):.2f}",
        f"yaw={math.degrees(yaw):.2f}",
    )

    for body_name in bodies:
        R = body_rotation(model, data, body_name)
        pos = body_position(model, data, body_name)
        results[body_name] = R

        print(f"\nbody: {body_name}")
        print("pos_w:", np.array2string(pos, precision=6, suppress_small=True))
        print("x_axis_w:", np.array2string(R[:, 0], precision=6, suppress_small=True))
        print("y_axis_w:", np.array2string(R[:, 1], precision=6, suppress_small=True))
        print("z_axis_w:", np.array2string(R[:, 2], precision=6, suppress_small=True))
        print("upright_dot:", f"{upright_dot(R):.8f}")
        print("orientation_l2_like:", f"{orientation_l2_like(R):.8f}")

    R_x1 = results["x1-body"]
    R_lumbar = results["link_lumbar_pitch"]

    # R_rel 表示从 x1-body 坐标系旋转到 link_lumbar_pitch 坐标系的相对旋转。
    R_rel = R_x1.T @ R_lumbar

    print("\nrelative: x1-body -> link_lumbar_pitch")
    print("relative_angle_deg:", f"{math.degrees(rotation_angle(R_rel)):.8f}")
    print("max_abs_axis_diff:", f"{np.max(np.abs(R_x1 - R_lumbar)):.8e}")


def main() -> None:
    """主函数。

    默认测试以下姿态：
    1. home_hover：正常悬空站姿
    2. yaw_90：只绕 z 轴旋转 90 度
    3. roll_10：只施加 10 度 roll
    4. pitch_10：只施加 10 度 pitch
    5. roll_10_pitch_10：同时施加 10 度 roll 和 10 度 pitch

    判断重点：
    home_hover 下 orientation_l2_like 应接近 0。
    yaw_90 下 orientation_l2_like 也应接近 0。
    roll_10 或 pitch_10 下 orientation_l2_like 应接近 sin(10 deg)^2。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to AgiBot X1 MJCF XML. Overrides AGIBOT_X1_XML.",
    )
    parser.add_argument(
        "--root-z",
        type=float,
        default=1.20,
        help="Floating base height used for the hover test.",
    )

    args = parser.parse_args()

    xml_path = resolve_xml_path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    print("xml:", xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    cases = [
        ("home_hover", 0.0, 0.0, 0.0),
        ("yaw_90", 0.0, 0.0, math.radians(90.0)),
        ("roll_10", math.radians(10.0), 0.0, 0.0),
        ("pitch_10", 0.0, math.radians(10.0), 0.0),
        ("roll_10_pitch_10", math.radians(10.0), math.radians(10.0), 0.0),
    ]

    for name, roll, pitch, yaw in cases:
        print_case(model, data, name, roll, pitch, yaw, args.root_z)


if __name__ == "__main__":
    main()