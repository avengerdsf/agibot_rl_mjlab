from __future__ import annotations

import socket
import struct
import time
from typing import Any

import numpy as np


class ImuClient:
    """把 IMU 数据通过 localhost TCP 发给外部桥接进程。

    发送格式
    --------
    每帧消息格式为:
    - 4 bytes: uint32 payload 长度
    - payload: 10 个 float32, 顺序如下
      [sim_time_s,
       ang_vel_x, ang_vel_y, ang_vel_z,
       lin_acc_x, lin_acc_y, lin_acc_z,
       quat_w, quat_x, quat_y, quat_z]

    如果上游没有姿态四元数，就发送单位四元数 [1, 0, 0, 0]。
    """

    _DEFAULT_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        send_hz: float = 200.0,
        connect_timeout: float = 0.05,
    ) -> None:
        self.host = host
        self.port = port
        self.send_hz = send_hz
        self.connect_timeout = connect_timeout

        self._sock: socket.socket | None = None
        self._last_send_time: float = -1.0

    def _connect(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        sock.connect((self.host, self.port))
        sock.settimeout(None)
        self._sock = sock

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def should_send(self) -> bool:
        now = time.monotonic()

        if self.send_hz <= 0.0:
            self._last_send_time = now
            return True

        period = 1.0 / self.send_hz
        if self._last_send_time < 0.0 or (now - self._last_send_time) >= period:
            self._last_send_time = now
            return True
        return False

    def send_imu32(
        self,
        sim_time_s: float,
        ang_vel_xyz: np.ndarray,
        lin_acc_xyz: np.ndarray,
        quat_wxyz: np.ndarray | None = None,
    ) -> None:
        """发送一帧 IMU 数据。"""
        ang = np.asarray(ang_vel_xyz, dtype=np.float32).reshape(3)
        acc = np.asarray(lin_acc_xyz, dtype=np.float32).reshape(3)
        quat = (
            self._DEFAULT_QUAT
            if quat_wxyz is None
            else np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        )

        payload_arr = np.concatenate(
            [
                np.asarray([sim_time_s], dtype=np.float32),
                ang,
                acc,
                quat,
            ],
            axis=0,
        )
        payload = payload_arr.tobytes()
        header = struct.pack("<I", len(payload))

        try:
            self._connect()
            assert self._sock is not None
            self._sock.sendall(header + payload)
        except Exception:
            self._close_socket()

    def close(self) -> None:
        self._close_socket()


class ImuSocketEnvWrapper:
    """兼容 mjlab 的轻量 env wrapper。

    只拦截 step()，其余属性透传给原始 env。
    """

    def __init__(
        self,
        env: Any,
        ang_vel_sensor_name: str = "robot/imu_ang_vel",
        lin_acc_sensor_name: str = "robot/imu_lin_acc",
        quat_sensor_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8766,
        send_hz: float = 200.0,
    ) -> None:
        self.env = env
        self.ang_vel_sensor_name = ang_vel_sensor_name
        self.lin_acc_sensor_name = lin_acc_sensor_name
        self.quat_sensor_name = quat_sensor_name
        self.client = ImuClient(
            host=host,
            port=port,
            send_hz=send_hz,
        )

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def _sensor_vec(self, sensor_name: str, env_id: int = 0) -> np.ndarray:
        sensor = self.unwrapped.scene[sensor_name]
        data = sensor.data
        if hasattr(data, "detach"):
            tensor = data
        elif hasattr(data, "data"):
            tensor = data.data
        else:
            tensor = data
        if hasattr(tensor, "detach"):
            return tensor[env_id].detach().cpu().numpy().astype(np.float32)
        return np.asarray(tensor[env_id], dtype=np.float32)

    def step(self, action):
        out = self.env.step(action)

        if self.client.should_send():
            ang_vel = self._sensor_vec(self.ang_vel_sensor_name, env_id=0)
            lin_acc = self._sensor_vec(self.lin_acc_sensor_name, env_id=0)
            quat = (
                None
                if self.quat_sensor_name is None
                else self._sensor_vec(self.quat_sensor_name, env_id=0)
            )

            base_env = self.unwrapped
            sim_time_s = float(base_env.episode_length_buf[0].item() * base_env.step_dt)
            self.client.send_imu32(
                sim_time_s=sim_time_s,
                ang_vel_xyz=ang_vel,
                lin_acc_xyz=lin_acc,
                quat_wxyz=quat,
            )

        return out

    def close(self):
        try:
            self.client.close()
        finally:
            self.env.close()
