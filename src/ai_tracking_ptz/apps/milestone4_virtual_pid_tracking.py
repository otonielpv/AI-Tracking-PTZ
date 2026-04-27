from __future__ import annotations

import argparse
import logging
import time

import cv2

from ai_tracking_ptz.camera.virtual_ptz import VirtualPTZCamera, VirtualPTZConfig
from ai_tracking_ptz.control.pid import PIDConfig, PIDController
from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.tracking.yolo_person_tracker import TrackedPerson, YoloPersonTracker, draw_tracks
from ai_tracking_ptz.video.file_stream import FileVideoStream


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 4: virtual PTZ PID tracking")
    parser.add_argument("--video-file", required=True, help="Local video file with people")
    parser.add_argument("--loop-video", action="store_true", help="Loop the local video when it reaches the end")
    parser.add_argument("--width", type=int, default=1280, help="Virtual camera output width")
    parser.add_argument("--height", type=int, default=720, help="Virtual camera output height")
    parser.add_argument("--max-zoom", type=float, default=4.0, help="Maximum virtual zoom")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config name")
    parser.add_argument("--imgsz", type=int, default=640, help="Internal inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-inference-fps", type=float, default=15.0, help="Inference rate limit")
    parser.add_argument("--device", default=None, help="Inference device. Example: cpu or cuda:0")
    parser.add_argument("--pid-pan-kp", type=float, default=1.2, help="PID Kp for pan")
    parser.add_argument("--pid-pan-ki", type=float, default=0.0, help="PID Ki for pan")
    parser.add_argument("--pid-pan-kd", type=float, default=0.25, help="PID Kd for pan")
    parser.add_argument("--pid-tilt-kp", type=float, default=1.0, help="PID Kp for tilt")
    parser.add_argument("--pid-tilt-ki", type=float, default=0.0, help="PID Ki for tilt")
    parser.add_argument("--pid-tilt-kd", type=float, default=0.2, help="PID Kd for tilt")
    parser.add_argument("--deadzone-x", type=float, default=0.05, help="Horizontal deadzone as normalized half-width")
    parser.add_argument("--deadzone-y", type=float, default=0.05, help="Vertical deadzone as normalized half-height")
    parser.add_argument("--window-name", default="AI Tracking PTZ - Milestone 4", help="OpenCV output window name")
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
    tracker = YoloPersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_fps=args.max_inference_fps,
    )
    pan_pid = PIDController(PIDConfig(kp=args.pid_pan_kp, ki=args.pid_pan_ki, kd=args.pid_pan_kd))
    tilt_pid = PIDController(PIDConfig(kp=args.pid_tilt_kp, ki=args.pid_tilt_ki, kd=args.pid_tilt_kd))

    last_ts = time.perf_counter()

    try:
        stream.start()
        LOGGER.info("Milestone 4 ready. Press q to exit.")
        while True:
            source_frame = stream.read()
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now

            if source_frame is None:
                waiting = build_waiting_frame(stream.is_connected)
                cv2.imshow(args.window_name, waiting)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            virtual_frame = camera.render(source_frame)
            tracks, ran_inference = tracker.maybe_track(virtual_frame)
            target = select_primary_target(tracks)
            pan_velocity, tilt_velocity, error_x, error_y = update_tracking_control(
                camera=camera,
                target=target,
                frame=virtual_frame,
                pan_pid=pan_pid,
                tilt_pid=tilt_pid,
                dt=dt,
                deadzone_x=args.deadzone_x,
                deadzone_y=args.deadzone_y,
            )
            camera.update(dt)

            annotated_output = virtual_frame.copy()
            draw_tracks(annotated_output, tracks)
            draw_target_marker(annotated_output, target)
            draw_deadzone(annotated_output, args.deadzone_x, args.deadzone_y)
            overlay_runtime_info(
                annotated_output,
                tracker=tracker,
                target=target,
                pan_velocity=pan_velocity,
                tilt_velocity=tilt_velocity,
                error_x=error_x,
                error_y=error_y,
                ran_inference=ran_inference,
            )

            source_preview = source_frame.copy()
            draw_source_viewport(source_preview, camera)

            cv2.imshow("Milestone 4 - Source", source_preview)
            cv2.imshow(args.window_name, annotated_output)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Milestone 4 virtual PID tracking failed.")
        return 1
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def update_tracking_control(
    camera: VirtualPTZCamera,
    target: TrackedPerson | None,
    frame,
    pan_pid: PIDController,
    tilt_pid: PIDController,
    dt: float,
    deadzone_x: float,
    deadzone_y: float,
) -> tuple[float, float, float, float]:
    if target is None:
        pan_pid.reset()
        tilt_pid.reset()
        camera.stop()
        return 0.0, 0.0, 0.0, 0.0

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = target.bbox_xyxy
    target_center_x = (x1 + x2) / 2.0
    target_center_y = (y1 + y2) / 2.0

    error_x = (target_center_x - (frame_width / 2.0)) / (frame_width / 2.0)
    error_y = (target_center_y - (frame_height / 2.0)) / (frame_height / 2.0)

    if abs(error_x) <= deadzone_x:
        error_x = 0.0
        pan_pid.reset()
    if abs(error_y) <= deadzone_y:
        error_y = 0.0
        tilt_pid.reset()

    pan_velocity = pan_pid.update(error_x, dt) if error_x != 0.0 else 0.0
    tilt_velocity = -tilt_pid.update(error_y, dt) if error_y != 0.0 else 0.0
    camera.continuous_move(pan_velocity=pan_velocity, tilt_velocity=tilt_velocity, zoom_velocity=0.0)
    return pan_velocity, tilt_velocity, error_x, error_y


def select_primary_target(tracks: list[TrackedPerson]) -> TrackedPerson | None:
    if not tracks:
        return None
    return max(tracks, key=lambda track: ((track.bbox_xyxy[2] - track.bbox_xyxy[0]) * (track.bbox_xyxy[3] - track.bbox_xyxy[1]), track.confidence))


def draw_target_marker(frame, target: TrackedPerson | None) -> None:
    if target is None:
        return
    x1, y1, x2, y2 = target.bbox_xyxy
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
    cv2.putText(frame, "target", (x1, max(20, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_deadzone(frame, deadzone_x: float, deadzone_y: float) -> None:
    frame_height, frame_width = frame.shape[:2]
    half_width = int(frame_width * deadzone_x)
    half_height = int(frame_height * deadzone_y)
    center_x = frame_width // 2
    center_y = frame_height // 2
    cv2.rectangle(
        frame,
        (center_x - half_width, center_y - half_height),
        (center_x + half_width, center_y + half_height),
        (0, 255, 0),
        2,
    )
    cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)


def draw_source_viewport(frame, camera: VirtualPTZCamera) -> None:
    x1, y1, x2, y2 = camera.describe_viewport(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(frame, "Virtual PTZ viewport", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)


def overlay_runtime_info(
    frame,
    tracker: YoloPersonTracker,
    target: TrackedPerson | None,
    pan_velocity: float,
    tilt_velocity: float,
    error_x: float,
    error_y: float,
    ran_inference: bool,
) -> None:
    tracker_stats = tracker.stats
    target_id = "none" if target is None or target.tracker_id is None else str(target.tracker_id)
    lines = [
        f"Target ID: {target_id}",
        f"Inference runs: {tracker_stats.inference_runs}",
        f"Inference FPS: {tracker_stats.effective_inference_fps:.1f}",
        f"Inference step: {'YOLO' if ran_inference else 'cached'}",
        f"Error X: {error_x:.3f} | Error Y: {error_y:.3f}",
        f"Pan vel: {pan_velocity:.3f} | Tilt vel: {tilt_velocity:.3f}",
    ]
    for index, line in enumerate(lines):
        y = 30 + (index * 28)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def build_waiting_frame(is_connected: bool):
    status = "Opening local video..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, "Loading tracker and PID...", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
