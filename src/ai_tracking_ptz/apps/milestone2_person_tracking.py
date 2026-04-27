from __future__ import annotations

import argparse
import logging
import time
from typing import Protocol

import cv2

from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.tracking.yolo_person_tracker import YoloPersonTracker, draw_tracks
from ai_tracking_ptz.video.file_stream import FileVideoStream
from ai_tracking_ptz.video.rtsp_stream import RTSPVideoStream


LOGGER = logging.getLogger(__name__)


class VideoStream(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def stats(self): ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self): ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 2: person detection and tracking")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rtsp-url", help="RTSP URL for the PTZ camera or test stream")
    source_group.add_argument("--video-file", help="Local video file for offline testing")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height")
    parser.add_argument(
        "--rtsp-transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport for FFmpeg backend",
    )
    parser.add_argument("--loop-video", action="store_true", help="Loop local video file when it reaches the end")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config name")
    parser.add_argument("--imgsz", type=int, default=640, help="Internal inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-inference-fps", type=float, default=15.0, help="Inference rate limit")
    parser.add_argument("--device", default=None, help="Inference device. Example: cpu or cuda:0")
    parser.add_argument("--window-name", default="AI Tracking PTZ - Milestone 2", help="OpenCV window name")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    stream = build_stream(args)
    tracker = YoloPersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_fps=args.max_inference_fps,
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

            tracks, ran_inference = tracker.maybe_track(frame)
            output_frame = frame.copy()
            draw_tracks(output_frame, tracks)
            overlay_runtime_info(output_frame, stream, tracker, len(tracks), ran_inference)
            cv2.imshow(args.window_name, output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Fatal error in milestone 2 viewer.")
        return 1
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def overlay_runtime_info(
    frame,
    stream: VideoStream,
    tracker: YoloPersonTracker,
    active_tracks: int,
    ran_inference: bool,
) -> None:
    stream_stats = stream.stats
    tracker_stats = tracker.stats
    lines = [
        f"Connected: {stream.is_connected}",
        f"Frames read: {stream_stats.frames_read}",
        f"Frames dropped: {stream_stats.frames_dropped}",
        f"Tracks: {active_tracks}",
        f"Inference runs: {tracker_stats.inference_runs}",
        f"Inference FPS: {tracker_stats.effective_inference_fps:.1f}",
        f"Inference step: {'YOLO' if ran_inference else 'cached'}",
    ]

    for index, line in enumerate(lines):
        y = 30 + (index * 28)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def _build_waiting_frame(is_connected: bool):
    status = "Connecting RTSP stream..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, "Loading YOLO tracker...", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def build_stream(args: argparse.Namespace) -> VideoStream:
    if args.video_file:
        return FileVideoStream(file_path=args.video_file, loop=args.loop_video)

    return RTSPVideoStream(
        rtsp_url=args.rtsp_url,
        width=args.width,
        height=args.height,
        rtsp_transport=args.rtsp_transport,
    )


if __name__ == "__main__":
    raise SystemExit(main())
