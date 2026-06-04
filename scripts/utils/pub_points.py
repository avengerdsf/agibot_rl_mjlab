from __future__ import annotations

import socket
import struct
import time
from typing import Any

import numpy as np


class Mid360PointCloudClient:
    """把点云通过 localhost TCP 发给 ROS2 桥接进程。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        send_hz: float = 15.0,
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

    def send_xyz32(
        self,
        sim_time_s: float,
        scan_period_s: float,
        sensor_pos_w: np.ndarray,
        sensor_quat_wxyz: np.ndarray,
        points_xyz: np.ndarray,
    ) -> None:
        """发送一帧点云。

        消息体格式:
        - 8 bytes: float64 sim_time_s
        - 8 bytes: float64 scan_period_s
        - 7 * float32: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        - 剩余: [N, 3] float32 点云
        """
        if points_xyz.size == 0:
            return

        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        pos = np.asarray(sensor_pos_w, dtype=np.float32).reshape(3)
        quat = np.asarray(sensor_quat_wxyz, dtype=np.float32).reshape(4)
        meta = struct.pack("<dd", float(sim_time_s), float(scan_period_s))
        pose = np.concatenate([pos, quat], axis=0).astype(np.float32).tobytes()
        payload = meta + pose + pts.tobytes()
        header = struct.pack("<I", len(payload))

        try:
            self._connect()
            assert self._sock is not None
            self._sock.sendall(header + payload)
        except Exception:
            self._close_socket()

    def close(self) -> None:
        self._close_socket()


class Mid360SocketEnvWrapper:
    """兼容 mjlab 的轻量 env wrapper。

    只拦截 step()，其余属性透传给原始 env。
    """

    def __init__(
        self,
        env: Any,
        sensor_name: str = "mid360",
        host: str = "127.0.0.1",
        port: int = 8765,
        send_hz: float = 15.0,
    ) -> None:
        self.env = env
        self.sensor_name = sensor_name
        self.client = Mid360PointCloudClient(
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

    def step(self, action):
        out = self.env.step(action)

        if self.client.should_send():
            sensor = self.unwrapped.scene[self.sensor_name]
            pts = sensor.get_current_pointcloud_lidar(env_id=0)
            pts_np = pts.detach().cpu().numpy().astype(np.float32)
            pos_w = sensor.data.pos_w[0].detach().cpu().numpy().astype(np.float32)
            quat_w = sensor.data.quat_w[0].detach().cpu().numpy().astype(np.float32)
            base_env = self.unwrapped
            sim_time_s = float(base_env.episode_length_buf[0].item() * base_env.step_dt)
            scan_period_s = 1.0 / self.client.send_hz if self.client.send_hz > 0.0 else float(base_env.step_dt)
            self.client.send_xyz32(
                sim_time_s=sim_time_s,
                scan_period_s=scan_period_s,
                sensor_pos_w=pos_w,
                sensor_quat_wxyz=quat_w,
                points_xyz=pts_np,
            )

        return out

    def close(self):
        try:
            self.client.close()
        finally:
            self.env.close()
