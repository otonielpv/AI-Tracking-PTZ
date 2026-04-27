from __future__ import annotations

import argparse
import logging
import socket

import cv2

from ai_tracking_ptz.camera.visca_over_ip import (
    DEFAULT_VISCA_PORT,
    PanDirection,
    TiltDirection,
    ViscaConnectionConfig,
    ViscaOverIPCamera,
    ZoomDirection,
)
from ai_tracking_ptz.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 3: VISCA keyboard test")
    parser.add_argument("--host", required=True, help="Camera IP or hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_VISCA_PORT, help="VISCA over IP port")
    parser.add_argument(
        "--transport",
        choices=["udp", "tcp"],
        default="udp",
        help="Transport for VISCA over IP. Start with udp for Sony-compatible VISCA cameras.",
    )
    parser.add_argument("--pan-speed", type=int, default=12, help="Pan speed 1-24")
    parser.add_argument("--tilt-speed", type=int, default=10, help="Tilt speed 1-20")
    parser.add_argument("--zoom-speed", type=int, default=3, help="Zoom speed 0-7")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    socket_type = socket.SOCK_DGRAM if args.transport == "udp" else socket.SOCK_STREAM
    camera = ViscaOverIPCamera(
        ViscaConnectionConfig(
            host=args.host,
            port=args.port,
            socket_type=socket_type,
        )
    )

    try:
        camera.connect()
        return run_keyboard_loop(camera, args.pan_speed, args.tilt_speed, args.zoom_speed)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("VISCA keyboard test failed.")
        return 1
    finally:
        try:
            camera.stop()
        except Exception:
            LOGGER.debug("Ignoring stop error during cleanup.", exc_info=True)
        camera.close()
        cv2.destroyAllWindows()


def run_keyboard_loop(camera: ViscaOverIPCamera, pan_speed: int, tilt_speed: int, zoom_speed: int) -> int:
    window_name = "AI Tracking PTZ - VISCA Test"
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
            camera.pan(PanDirection.LEFT, pan_speed)
        elif key == ord("d"):
            camera.pan(PanDirection.RIGHT, pan_speed)
        elif key == ord("w"):
            camera.tilt(TiltDirection.UP, tilt_speed)
        elif key == ord("s"):
            camera.tilt(TiltDirection.DOWN, tilt_speed)
        elif key == ord("z"):
            camera.zoom(ZoomDirection.WIDE, zoom_speed)
        elif key == ord("x"):
            camera.zoom(ZoomDirection.TELE, zoom_speed)
        elif key == ord(" "):
            camera.stop()


def build_ui_frame(camera: ViscaOverIPCamera, pan_speed: int, tilt_speed: int, zoom_speed: int):
    frame = cv2.UMat(540, 960, cv2.CV_8UC3).get()
    frame[:] = (16, 16, 16)

    lines = [
        "Hito 3 - VISCA over IP keyboard test",
        f"Connected: {camera.is_connected}",
        f"Pan speed: {pan_speed} | Tilt speed: {tilt_speed} | Zoom speed: {zoom_speed}",
        "W: tilt up | S: tilt down",
        "A: pan left | D: pan right",
        "Z: zoom wide | X: zoom tele",
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
