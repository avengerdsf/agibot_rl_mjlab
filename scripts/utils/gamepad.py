"""
Pure-Python Linux joystick reader for /dev/input/jsX devices.

Axis/button mappings mirror simulate/src/physics_joystick.h exactly:
  XBox  : lx=axis0, ly=-axis1, rx=axis3, ry=-axis4, LT=axis2>0, RT=axis5>0
  Switch: lx=axis0, ly=-axis1, rx=axis2, ry=-axis3, LT=axis5>0, RT=axis4>0

Twist output (lin_vel_x, lin_vel_y, ang_vel_z):
  lin_vel_x  =  ly  (left stick forward/back)
  lin_vel_y  = -lx  (left stick left/right, sign matches obs[1]=-joystick->lx())
  ang_vel_z  = -rx  (right stick left/right, negated → CCW positive)
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Linux joystick event: 8 bytes  (time_ms u32, value i16, type u8, number u8)
_JS_EVENT_FMT = "IhBB"
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)

_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS   = 0x02
_JS_EVENT_INIT   = 0x80  # synthetic init event flag


# ---------------------------------------------------------------------------
# Button index maps  (matches physics_joystick.h)
# ---------------------------------------------------------------------------

_XBOX_BUTTONS: dict[str, int] = {
    "A": 0, "B": 1, "X": 2, "Y": 3,
    "LB": 4, "RB": 5,
    "back": 6, "start": 7,
}

_SWITCH_BUTTONS: dict[str, int] = {
    "A": 0, "B": 1, "X": 3, "Y": 4,
    "LB": 6, "RB": 7,
    "back": 10, "start": 11,
}

# D-pad is encoded as axes 6 (left/right) and 7 (up/down) on both layouts.
_DPAD_AXIS_X = 6   # <0 → left,  >0 → right
_DPAD_AXIS_Y = 7   # <0 → up,    >0 → down


@dataclass
class GamepadConfig:
    device_path: str = "/dev/input/js0"
    gamepad_type: Literal["xbox", "switch"] = "xbox"
    # Raw axis range is  -(2^(axis_bits-1)) … +(2^(axis_bits-1)-1)
    axis_bits: int = 15
    deadzone: float = 0.05   # normalised, applied after scaling
    scale_x: float = 1.0     # multiplier for lin_vel_x / lin_vel_y
    scale_yaw: float = 1.0   # multiplier for ang_vel_z
    reconnect_interval: float = 2.0  # seconds between reconnect attempts


@dataclass
class TwistCommand:
    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0


@dataclass
class ButtonState:
    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False
    LB: bool = False
    RB: bool = False
    LT: bool = False   # trigger treated as digital (axis > 0)
    RT: bool = False
    back: bool = False
    start: bool = False
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False


class GamepadReader:
    """
    Background-thread Linux joystick reader.

    Usage::

        gp = GamepadReader(GamepadConfig(gamepad_type="xbox"))
        gp.start()
        ...
        cmd = gp.read_command()   # TwistCommand, thread-safe
        btn = gp.read_buttons()   # ButtonState, thread-safe
        ...
        gp.stop()
    """

    def __init__(self, cfg: GamepadConfig | None = None) -> None:
        self.cfg = cfg or GamepadConfig()
        self._max_value: int = 1 << (self.cfg.axis_bits - 1)

        # Raw normalised axis values [-1, 1] (before deadzone/scale)
        self._axes: dict[int, float] = {}
        # Raw button states
        self._buttons: dict[int, bool] = {}

        self._lock = threading.Lock()
        self._twist = TwistCommand()
        self._btn_state = ButtonState()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()  # set = connected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gamepad-reader")
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def read_command(self) -> TwistCommand:
        """Return the latest normalised twist command (thread-safe copy)."""
        with self._lock:
            return TwistCommand(
                lin_vel_x=self._twist.lin_vel_x,
                lin_vel_y=self._twist.lin_vel_y,
                ang_vel_z=self._twist.ang_vel_z,
            )

    def read_buttons(self) -> ButtonState:
        """Return a snapshot of the current button state (thread-safe copy)."""
        with self._lock:
            import copy
            return copy.copy(self._btn_state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            path = Path(self.cfg.device_path)
            if not path.exists():
                self._connected.clear()
                self._stop_event.wait(self.cfg.reconnect_interval)
                continue

            try:
                with open(path, "rb") as fd:
                    self._connected.set()
                    self._read_loop(fd)
            except OSError:
                pass

            self._connected.clear()
            # Zero out outputs on disconnect
            with self._lock:
                self._twist = TwistCommand()
                self._btn_state = ButtonState()

            if not self._stop_event.is_set():
                self._stop_event.wait(self.cfg.reconnect_interval)

    def _read_loop(self, fd) -> None:
        import select
        while not self._stop_event.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            raw = fd.read(_JS_EVENT_SIZE)
            if len(raw) < _JS_EVENT_SIZE:
                break  # device closed / unplugged
            self._handle_event(raw)

    def _handle_event(self, raw: bytes) -> None:
        _time_ms, value, ev_type, number = struct.unpack(_JS_EVENT_FMT, raw)
        ev_type &= ~_JS_EVENT_INIT  # strip init flag

        if ev_type == _JS_EVENT_AXIS:
            norm = value / self._max_value
            norm = max(-1.0, min(1.0, norm))
            self._axes[number] = norm
        elif ev_type == _JS_EVENT_BUTTON:
            self._buttons[number] = bool(value)
        else:
            return

        self._recompute()

    def _recompute(self) -> None:
        a = self._axes
        b = self._buttons
        t = self.cfg.gamepad_type

        def ax(idx: int) -> float:
            return a.get(idx, 0.0)

        def btn(idx: int) -> bool:
            return b.get(idx, False)

        if t == "xbox":
            lx  =  ax(0)
            ly  = -ax(1)
            rx  =  ax(3)
            # ry  = -ax(4)  # unused for twist
            lt  = ax(2) > 0
            rt  = ax(5) > 0
            bmap = _XBOX_BUTTONS
        else:  # switch
            lx  =  ax(0)
            ly  = -ax(1)
            rx  =  ax(2)
            # ry  = -ax(3)
            lt  = ax(5) > 0
            rt  = ax(4) > 0
            bmap = _SWITCH_BUTTONS

        # D-pad (shared between layouts)
        dpad_x = ax(_DPAD_AXIS_X)
        dpad_y = ax(_DPAD_AXIS_Y)

        # Apply deadzone
        lx  = self._dz(lx)
        ly  = self._dz(ly)
        rx  = self._dz(rx)

        # Twist: forward=ly, strafe=-lx (matches obs[1]=-joystick->lx()), yaw=-rx
        twist = TwistCommand(
            lin_vel_x =  ly  * self.cfg.scale_x,
            lin_vel_y = -lx  * self.cfg.scale_x,
            ang_vel_z = -rx  * self.cfg.scale_yaw,
        )

        buttons = ButtonState(
            A     = btn(bmap["A"]),
            B     = btn(bmap["B"]),
            X     = btn(bmap["X"]),
            Y     = btn(bmap["Y"]),
            LB    = btn(bmap["LB"]),
            RB    = btn(bmap["RB"]),
            LT    = lt,
            RT    = rt,
            back  = btn(bmap["back"]),
            start = btn(bmap["start"]),
            up    = dpad_y < -0.5,
            down  = dpad_y >  0.5,
            left  = dpad_x < -0.5,
            right = dpad_x >  0.5,
        )

        with self._lock:
            self._twist = twist
            self._btn_state = buttons

    def _dz(self, v: float) -> float:
        """Apply deadzone: values within ±deadzone are zeroed."""
        dz = self.cfg.deadzone
        if abs(v) < dz:
            return 0.0
        # Rescale so output starts at 0 just outside the deadzone
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - dz) / (1.0 - dz)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_gamepad(
    device: str = "/dev/input/js0",
    gamepad_type: Literal["xbox", "switch"] = "xbox",
    deadzone: float = 0.05,
    scale_x: float = 1.0,
    scale_yaw: float = 1.0,
    axis_bits: int = 15,
    autostart: bool = True,
) -> GamepadReader:
    cfg = GamepadConfig(
        device_path=device,
        gamepad_type=gamepad_type,
        axis_bits=axis_bits,
        deadzone=deadzone,
        scale_x=scale_x,
        scale_yaw=scale_yaw,
    )
    gp = GamepadReader(cfg)
    if autostart:
        gp.start()
    return gp


# ---------------------------------------------------------------------------
# CLI smoke-test:  python -m scripts.utils.gamepad
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/input/js0"
    gtype  = sys.argv[2] if len(sys.argv) > 2 else "xbox"

    print(f"Reading {device} as {gtype}. Ctrl-C to quit.")
    gp = make_gamepad(device=device, gamepad_type=gtype)  # type: ignore[arg-type]

    try:
        while True:
            cmd = gp.read_command()
            btn = gp.read_buttons()
            connected = "OK" if gp.is_connected() else "DISCONNECTED"
            print(
                f"\r[{connected}]  "
                f"vx={cmd.lin_vel_x:+.3f}  vy={cmd.lin_vel_y:+.3f}  wz={cmd.ang_vel_z:+.3f}  "
                f"A={int(btn.A)} B={int(btn.B)} start={int(btn.start)} back={int(btn.back)}",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        gp.stop()
