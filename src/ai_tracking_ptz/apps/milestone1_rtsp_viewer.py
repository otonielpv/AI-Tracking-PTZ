from __future__ import annotations

import argparse
import logging
import time

import cv2

from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.video.rtsp_stream import RTSPVideoStream


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 1: threaded RTSP viewer")
    parser.add_argument("--rtsp-url", required=True, help="RTSP URL for the PTZ camera")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height")
    parser.add_argument(
        "--rtsp-transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport for FFmpeg backend",
    )
    parser.add_argument("--window-name", default="AI Tracking PTZ - RTSP", help="OpenCV window name")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    stream = RTSPVideoStream(
        rtsp_url=args.rtsp_url,
        width=args.width,
        height=args.height,
        rtsp_transport=args.rtsp_transport,
    )

    try:
        stream.start()
        LOGGER.info("Press 'q' to exit.")

        while True:
            frame = stream.read()
            if frame is None:
                waiting_frame = _build_waiting_frame(stream.is_connected)
                cv2.imshow(args.window_name, waiting_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.01)
                continue

            overlay_runtime_info(frame, stream)
            cv2.imshow(args.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Fatal error in milestone 1 viewer.")
        return 1
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def overlay_runtime_info(frame, stream: RTSPVideoStream) -> None:
    stats = stream.stats
    lines = [
        f"Connected: {stream.is_connected}",
        f"Frames read: {stats.frames_read}",
        f"Frames dropped: {stats.frames_dropped}",
    ]

    for index, line in enumerate(lines):
        y = 30 + (index * 28)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def _build_waiting_frame(is_connected: bool):
    status = "Connecting RTSP stream..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
