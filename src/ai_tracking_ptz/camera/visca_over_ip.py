from __future__ import annotations

import logging
import socket
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional


LOGGER = logging.getLogger(__name__)

DEFAULT_VISCA_PORT = 52381
VISCA_HEADER_TYPE_COMMAND = 0x0100
VISCA_HEADER_TYPE_SEQUENCE_RESET = 0x0200


class PanDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


class TiltDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    STOP = "stop"


class ZoomDirection(str, Enum):
    TELE = "tele"
    WIDE = "wide"
    STOP = "stop"


@dataclass(slots=True)
class ViscaConnectionConfig:
    host: str
    port: int = DEFAULT_VISCA_PORT
    timeout_s: float = 1.0
    socket_type: int = socket.SOCK_DGRAM


class ViscaOverIPCamera:
    def __init__(self, config: ViscaConnectionConfig) -> None:
        self.config = config
        self._socket: Optional[socket.socket] = None
        self._sequence_number = 1

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self._socket is not None:
            return

        try:
            visca_socket = socket.socket(socket.AF_INET, self.config.socket_type)
            visca_socket.settimeout(self.config.timeout_s)
            visca_socket.connect((self.config.host, self.config.port))
            self._socket = visca_socket
            self.reset_sequence_number()
            LOGGER.info(
                "Connected to VISCA camera at %s:%s over %s",
                self.config.host,
                self.config.port,
                "UDP" if self.config.socket_type == socket.SOCK_DGRAM else "TCP",
            )
        except Exception:
            self.close()
            LOGGER.exception("Unable to connect to VISCA camera.")
            raise

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def reset_sequence_number(self) -> None:
        self._sequence_number = 1
        self._send_packet(b"\x01", packet_type=VISCA_HEADER_TYPE_SEQUENCE_RESET)

    def pan(self, direction: PanDirection | str, speed: int) -> None:
        pan_direction = PanDirection(direction)
        if pan_direction == PanDirection.STOP:
            self.stop_pan_tilt()
            return
        self.pan_tilt(pan_direction=pan_direction, tilt_direction=TiltDirection.STOP, pan_speed=speed, tilt_speed=1)

    def tilt(self, direction: TiltDirection | str, speed: int) -> None:
        tilt_direction = TiltDirection(direction)
        if tilt_direction == TiltDirection.STOP:
            self.stop_pan_tilt()
            return
        self.pan_tilt(pan_direction=PanDirection.STOP, tilt_direction=tilt_direction, pan_speed=1, tilt_speed=speed)

    def pan_tilt(
        self,
        pan_direction: PanDirection | str,
        tilt_direction: TiltDirection | str,
        pan_speed: int,
        tilt_speed: int,
    ) -> None:
        pan_speed_byte = self._clamp_pan_speed(pan_speed)
        tilt_speed_byte = self._clamp_tilt_speed(tilt_speed)
        payload = bytes(
            [
                0x81,
                0x01,
                0x06,
                0x01,
                pan_speed_byte,
                tilt_speed_byte,
                self._encode_pan_direction(PanDirection(pan_direction)),
                self._encode_tilt_direction(TiltDirection(tilt_direction)),
                0xFF,
            ]
        )
        self._send_visca_command(payload)

    def zoom(self, direction: ZoomDirection | str, speed: int = 3) -> None:
        zoom_direction = ZoomDirection(direction)
        if zoom_direction == ZoomDirection.STOP:
            payload = b"\x81\x01\x04\x07\x00\xFF"
            self._send_visca_command(payload)
            return

        zoom_speed = self._clamp_zoom_speed(speed)
        command = 0x20 + zoom_speed if zoom_direction == ZoomDirection.TELE else 0x30 + zoom_speed
        payload = bytes([0x81, 0x01, 0x04, 0x07, command, 0xFF])
        self._send_visca_command(payload)

    def stop_pan_tilt(self) -> None:
        self.pan_tilt(
            pan_direction=PanDirection.STOP,
            tilt_direction=TiltDirection.STOP,
            pan_speed=1,
            tilt_speed=1,
        )

    def stop(self) -> None:
        self.stop_pan_tilt()
        self.zoom(ZoomDirection.STOP)

    def _send_visca_command(self, payload: bytes) -> None:
        self._send_packet(payload, packet_type=VISCA_HEADER_TYPE_COMMAND)

    def _send_packet(self, payload: bytes, packet_type: int) -> None:
        if self._socket is None:
            raise RuntimeError("VISCA camera is not connected.")

        header = struct.pack(">HHI", packet_type, len(payload), self._sequence_number)
        packet = header + payload
        self._socket.send(packet)
        LOGGER.debug("Sent VISCA packet seq=%s payload=%s", self._sequence_number, payload.hex(" "))
        if packet_type == VISCA_HEADER_TYPE_COMMAND:
            self._sequence_number += 1

    @staticmethod
    def _encode_pan_direction(direction: PanDirection) -> int:
        mapping = {
            PanDirection.LEFT: 0x01,
            PanDirection.RIGHT: 0x02,
            PanDirection.STOP: 0x03,
        }
        return mapping[direction]

    @staticmethod
    def _encode_tilt_direction(direction: TiltDirection) -> int:
        mapping = {
            TiltDirection.UP: 0x01,
            TiltDirection.DOWN: 0x02,
            TiltDirection.STOP: 0x03,
        }
        return mapping[direction]

    @staticmethod
    def _clamp_pan_speed(speed: int) -> int:
        return max(1, min(24, int(speed)))

    @staticmethod
    def _clamp_tilt_speed(speed: int) -> int:
        return max(1, min(20, int(speed)))

    @staticmethod
    def _clamp_zoom_speed(speed: int) -> int:
        if speed <= 0:
            return 0
        if speed > 7:
            return 7
        return int(speed)
