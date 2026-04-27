from __future__ import annotations

import argparse
import logging

import cv2

from ai_tracking_ptz.camera.onvif_ptz import DEFAULT_ONVIF_PORT, OnvifConnectionConfig, OnvifPTZCamera
from ai_tracking_ptz.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 3: ONVIF keyboard test")
    parser.add_argument("--host", required=True, help="Camera IP or hostname")
    parser.add_argument("--username", required=True, help="ONVIF username")
    parser.add_argument("--password", required=True, help="ONVIF password")
    parser.add_argument("--port", type=int, default=DEFAULT_ONVIF_PORT, help="ONVIF service port")
    parser.add_argument("--wsdl-dir", default=None, help="Optional path to ONVIF WSDL directory")
    parser.add_argument("--pan-speed", type=float, default=0.5, help="Pan velocity 0.0-1.0")
    parser.add_argument("--tilt-speed", type=float, default=0.5, help="Tilt velocity 0.0-1.0")
    parser.add_argument("--zoom-speed", type=float, default=0.3, help="Zoom velocity 0.0-1.0")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    camera = OnvifPTZCamera(
        OnvifConnectionConfig(
            host=args.host,
            username=args.username,
            password=args.password,
            port=args.port,
            wsdl_dir=args.wsdl_dir,
        )
    )

    try:
        camera.connect()
        return run_keyboard_loop(camera, args.pan_speed, args.tilt_speed, args.zoom_speed)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("ONVIF keyboard test failed.")
        return 1
    finally:
        try:
            camera.stop()
        except Exception:
            LOGGER.debug("Ignoring stop error during cleanup.", exc_info=True)
        camera.close()
        cv2.destroyAllWindows()


def run_keyboard_loop(camera: OnvifPTZCamera, pan_speed: float, tilt_speed: float, zoom_speed: float) -> int:
    window_name = "AI Tracking PTZ - ONVIF Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    LOGGER.info("Keyboard test ready. Use W/A/S/D for tilt/pan, Z/X for zoom, Space to stop, Q to quit.")

    while True:
        frame = build_ui_frame(camera, pan_speed, tilt_speed, zoom_speed)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            return 0
        if key == ord("a"):
            camera.pan(-pan_speed)
        elif key == ord("d"):
            camera.pan(pan_speed)
        elif key == ord("w"):
            camera.tilt(tilt_speed)
        elif key == ord("s"):
            camera.tilt(-tilt_speed)
        elif key == ord("z"):
            camera.zoom(-zoom_speed)
        elif key == ord("x"):
            camera.zoom(zoom_speed)
        elif key == ord(" "):
            camera.stop()


def build_ui_frame(camera: OnvifPTZCamera, pan_speed: float, tilt_speed: float, zoom_speed: float):
    frame = cv2.UMat(540, 960, cv2.CV_8UC3).get()
    frame[:] = (16, 16, 16)

    lines = [
        "Hito 3 - ONVIF PTZ keyboard test",
        f"Connected: {camera.is_connected}",
        f"Pan speed: {pan_speed:.2f} | Tilt speed: {tilt_speed:.2f} | Zoom speed: {zoom_speed:.2f}",
        "W: tilt up | S: tilt down",
        "A: pan left | D: pan right",
        "Z: zoom out | X: zoom in",
        "Space: stop movement and zoom",
        "Q or Esc: exit",
    ]

    for index, line in enumerate(lines):
        y = 60 + (index * 48)
        color = (0, 220, 255) if index == 0 else (240, 240, 240)
        thickness = 2 if index == 0 else 1
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
