from __future__ import annotations

import argparse
import logging
import time

import cv2

from ai_tracking_ptz.camera.virtual_ptz import VirtualPTZCamera, VirtualPTZConfig
from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.video.file_stream import FileVideoStream


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual PTZ keyboard test")
    parser.add_argument("--video-file", required=True, help="Local video file for PTZ emulation")
    parser.add_argument("--loop-video", action="store_true", help="Loop the local video when it reaches the end")
    parser.add_argument("--width", type=int, default=1280, help="Virtual camera output width")
    parser.add_argument("--height", type=int, default=720, help="Virtual camera output height")
    parser.add_argument("--max-zoom", type=float, default=4.0, help="Maximum virtual zoom")
    parser.add_argument("--pan-speed", type=float, default=0.5, help="Pan velocity 0.0-1.0")
    parser.add_argument("--tilt-speed", type=float, default=0.5, help="Tilt velocity 0.0-1.0")
    parser.add_argument("--zoom-speed", type=float, default=0.4, help="Zoom velocity 0.0-1.0")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    stream = FileVideoStream(file_path=args.video_file, loop=args.loop_video)
    camera = VirtualPTZCamera(
        VirtualPTZConfig(
            output_width=args.width,
            output_height=args.height,
            max_zoom=args.max_zoom,
        )
    )

    last_ts = time.perf_counter()

    try:
        stream.start()
        LOGGER.info("Virtual PTZ ready. Use W/A/S/D for tilt/pan, Z/X for zoom, Space to stop, Q to quit.")
        while True:
            frame = stream.read()
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now

            if frame is None:
                waiting = build_waiting_frame(stream.is_connected)
                cv2.imshow("Virtual PTZ Output", waiting)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            handle_key(camera, cv2.waitKey(1) & 0xFF, args.pan_speed, args.tilt_speed, args.zoom_speed)
            camera.update(dt)

            virtual_frame = camera.render(frame)
            source_preview = frame.copy()
            draw_viewport_preview(source_preview, camera)

            cv2.imshow("Virtual PTZ Source", source_preview)
            cv2.imshow("Virtual PTZ Output", virtual_frame)

        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Virtual PTZ test failed.")
        return 1
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def handle_key(camera: VirtualPTZCamera, key: int, pan_speed: float, tilt_speed: float, zoom_speed: float) -> None:
    if key == 255:
        return
    if key in (ord("q"), 27):
        raise KeyboardInterrupt
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


def draw_viewport_preview(frame, camera: VirtualPTZCamera) -> None:
    x1, y1, x2, y2 = camera.describe_viewport(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(frame, "Virtual PTZ viewport", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)


def build_waiting_frame(is_connected: bool):
    status = "Opening local video..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
