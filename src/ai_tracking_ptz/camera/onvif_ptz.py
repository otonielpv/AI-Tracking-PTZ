from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from onvif import ONVIFCamera


LOGGER = logging.getLogger(__name__)

DEFAULT_ONVIF_PORT = 80


@dataclass(slots=True)
class OnvifConnectionConfig:
    host: str
    username: str
    password: str
    port: int = DEFAULT_ONVIF_PORT
    wsdl_dir: Optional[str] = None


class OnvifPTZCamera:
    def __init__(self, config: OnvifConnectionConfig) -> None:
        self.config = config
        self._camera: Optional[ONVIFCamera] = None
        self._media_service = None
        self._ptz_service = None
        self._profile = None
        self._profile_token: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._ptz_service is not None and self._profile_token is not None

    def connect(self) -> None:
        if self.is_connected:
            return

        try:
            camera = ONVIFCamera(
                self.config.host,
                self.config.port,
                self.config.username,
                self.config.password,
                self.config.wsdl_dir,
            )
            media_service = camera.create_media_service()
            ptz_service = camera.create_ptz_service()
            profiles = media_service.GetProfiles()
            if not profiles:
                raise RuntimeError("ONVIF camera returned no media profiles.")

            self._camera = camera
            self._media_service = media_service
            self._ptz_service = ptz_service
            self._profile = profiles[0]
            self._profile_token = self._profile.token

            LOGGER.info(
                "Connected to ONVIF camera at %s:%s using profile token %s",
                self.config.host,
                self.config.port,
                self._profile_token,
            )
        except Exception:
            self.close()
            LOGGER.exception("Unable to connect to ONVIF camera.")
            raise

    def close(self) -> None:
        self._camera = None
        self._media_service = None
        self._ptz_service = None
        self._profile = None
        self._profile_token = None

    def continuous_move(self, pan_velocity: float = 0.0, tilt_velocity: float = 0.0, zoom_velocity: float = 0.0) -> None:
        ptz_service = self._require_ptz_service()
        request = ptz_service.create_type("ContinuousMove")
        request.ProfileToken = self._profile_token
        request.Velocity = {
            "PanTilt": {"x": self._clamp_unit(pan_velocity), "y": self._clamp_unit(tilt_velocity)},
            "Zoom": {"x": self._clamp_unit(zoom_velocity)},
        }
        ptz_service.ContinuousMove(request)

    def pan(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=velocity, tilt_velocity=0.0, zoom_velocity=0.0)

    def tilt(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=0.0, tilt_velocity=velocity, zoom_velocity=0.0)

    def zoom(self, velocity: float) -> None:
        self.continuous_move(pan_velocity=0.0, tilt_velocity=0.0, zoom_velocity=velocity)

    def stop(self, stop_pan_tilt: bool = True, stop_zoom: bool = True) -> None:
        ptz_service = self._require_ptz_service()
        request = ptz_service.create_type("Stop")
        request.ProfileToken = self._profile_token
        request.PanTilt = stop_pan_tilt
        request.Zoom = stop_zoom
        ptz_service.Stop(request)

    def _require_ptz_service(self):
        if not self.is_connected:
            raise RuntimeError("ONVIF camera is not connected.")
        return self._ptz_service

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))
