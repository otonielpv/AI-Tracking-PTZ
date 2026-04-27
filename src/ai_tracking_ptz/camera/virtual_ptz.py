from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VirtualPTZConfig:
    output_width: int = 1280
    output_height: int = 720
    min_zoom: float = 1.0
    max_zoom: float = 4.0
    max_pan_rate: float = 0.6
    max_tilt_rate: float = 0.6
    max_zoom_rate: float = 1.25


@dataclass(slots=True)
class VirtualPTZState:
    center_x: float = 0.5
    center_y: float = 0.5
    zoom: float = 1.0
    pan_velocity: float = 0.0
    tilt_velocity: float = 0.0
    zoom_velocity: float = 0.0


class VirtualPTZCamera:
    def __init__(self, config: VirtualPTZConfig | None = None) -> None:
        self.config = config or VirtualPTZConfig()
        self.state = VirtualPTZState()

    @property
    def is_connected(self) -> bool:
        return True

    def continuous_move(self, pan_velocity: float = 0.0, tilt_velocity: float = 0.0, zoom_velocity: float = 0.0) -> None:
        self.state.pan_velocity = self._clamp_unit(pan_velocity)
        self.state.tilt_velocity = self._clamp_unit(tilt_velocity)
        self.state.zoom_velocity = self._clamp_unit(zoom_velocity)

    def pan(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=velocity, tilt_velocity=0.0, zoom_velocity=0.0)

    def tilt(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=0.0, tilt_velocity=velocity, zoom_velocity=0.0)

    def zoom(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=0.0, tilt_velocity=0.0, zoom_velocity=velocity)

    def stop(self) -> None:
        self.continuous_move(0.0, 0.0, 0.0)

    def update(self, dt: float) -> None:
        if dt <= 0:
            return

        self.state.center_x = self._clamp_position(
            self.state.center_x + (self.state.pan_velocity * self.config.max_pan_rate * dt)
        )
        self.state.center_y = self._clamp_position(
            self.state.center_y - (self.state.tilt_velocity * self.config.max_tilt_rate * dt)
        )
        zoom_delta = self.state.zoom_velocity * self.config.max_zoom_rate * dt
        self.state.zoom = max(self.config.min_zoom, min(self.config.max_zoom, self.state.zoom + zoom_delta))

    def render(self, source_frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = source_frame.shape[:2]
        crop_width, crop_height = self._compute_crop_size(frame_width, frame_height)
        center_x_px = int(self.state.center_x * frame_width)
        center_y_px = int(self.state.center_y * frame_height)

        x1 = center_x_px - (crop_width // 2)
        y1 = center_y_px - (crop_height // 2)
        x1 = max(0, min(frame_width - crop_width, x1))
        y1 = max(0, min(frame_height - crop_height, y1))
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        crop = source_frame[y1:y2, x1:x2]
        output = cv2.resize(crop, (self.config.output_width, self.config.output_height), interpolation=cv2.INTER_LINEAR)

        self._draw_overlay(output)
        return output

    def describe_viewport(self, source_frame: np.ndarray) -> tuple[int, int, int, int]:
        frame_height, frame_width = source_frame.shape[:2]
        crop_width, crop_height = self._compute_crop_size(frame_width, frame_height)
        center_x_px = int(self.state.center_x * frame_width)
        center_y_px = int(self.state.center_y * frame_height)
        x1 = max(0, min(frame_width - crop_width, center_x_px - (crop_width // 2)))
        y1 = max(0, min(frame_height - crop_height, center_y_px - (crop_height // 2)))
        return x1, y1, x1 + crop_width, y1 + crop_height

    def _compute_crop_size(self, frame_width: int, frame_height: int) -> tuple[int, int]:
        crop_width = max(32, int(frame_width / self.state.zoom))
        crop_height = max(32, int(frame_height / self.state.zoom))

        target_aspect = self.config.output_width / self.config.output_height
        current_aspect = crop_width / crop_height

        if current_aspect > target_aspect:
            crop_width = max(32, int(crop_height * target_aspect))
        else:
            crop_height = max(32, int(crop_width / target_aspect))

        crop_width = min(frame_width, crop_width)
        crop_height = min(frame_height, crop_height)
        return crop_width, crop_height

    def _draw_overlay(self, frame: np.ndarray) -> None:
        lines = [
            "Virtual PTZ",
            f"Center: ({self.state.center_x:.2f}, {self.state.center_y:.2f})",
            f"Zoom: {self.state.zoom:.2f}x",
            f"Vel: pan {self.state.pan_velocity:.2f} | tilt {self.state.tilt_velocity:.2f} | zoom {self.state.zoom_velocity:.2f}",
        ]
        for index, line in enumerate(lines):
            y = 30 + (index * 26)
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _clamp_position(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
