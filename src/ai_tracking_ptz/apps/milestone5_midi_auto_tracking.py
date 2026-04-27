from __future__ import annotations

import argparse
import logging
import time

import cv2

from ai_tracking_ptz.camera.virtual_ptz import VirtualPTZCamera, VirtualPTZConfig
from ai_tracking_ptz.control.pid import PIDConfig, PIDController
from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.midi.controller import MidiMappingConfig, MidiTrackingController
from ai_tracking_ptz.tracking.target_selector import AutoTargetSelector, TargetSelectorConfig
from ai_tracking_ptz.tracking.yolo_person_tracker import TrackedPerson, YoloPersonTracker, draw_tracks
from ai_tracking_ptz.video.file_stream import FileVideoStream


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 5: MIDI-controlled automatic AI tracking")
    parser.add_argument("--video-file", required=True, help="Local video file with people")
    parser.add_argument("--loop-video", action="store_true", help="Loop the local video when it reaches the end")
    parser.add_argument("--width", type=int, default=1280, help="Virtual camera output width")
    parser.add_argument("--height", type=int, default=720, help="Virtual camera output height")
    parser.add_argument("--max-zoom", type=float, default=4.0, help="Maximum virtual zoom")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path. Can be .pt or .engine")
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
    parser.add_argument("--midi-input-name", default=None, help="MIDI input port name")
    parser.add_argument("--midi-channel", type=int, default=0, help="MIDI channel 0-15")
    parser.add_argument("--midi-toggle-note", type=int, default=60, help="MIDI note for toggle tracking")
    parser.add_argument("--midi-enable-note", type=int, default=61, help="MIDI note for enable tracking")
    parser.add_argument("--midi-disable-note", type=int, default=62, help="MIDI note for disable tracking")
    parser.add_argument("--midi-reacquire-note", type=int, default=63, help="MIDI note for reacquire target")
    parser.add_argument("--start-enabled", action="store_true", help="Start with tracking enabled")
    parser.add_argument("--selector-area-weight", type=float, default=0.45, help="Auto-target area weight")
    parser.add_argument("--selector-center-weight", type=float, default=0.25, help="Auto-target center weight")
    parser.add_argument("--selector-confidence-weight", type=float, default=0.10, help="Auto-target confidence weight")
    parser.add_argument("--selector-persistence-weight", type=float, default=0.20, help="Auto-target persistence weight")
    parser.add_argument("--selector-target-bonus", type=float, default=0.20, help="Auto-target lock bonus")
    parser.add_argument("--window-name", default="AI Tracking PTZ - Milestone 5", help="OpenCV output window name")
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
    selector = AutoTargetSelector(
        TargetSelectorConfig(
            area_weight=args.selector_area_weight,
            center_weight=args.selector_center_weight,
            confidence_weight=args.selector_confidence_weight,
            persistence_weight=args.selector_persistence_weight,
            current_target_bonus=args.selector_target_bonus,
        )
    )
    midi_controller = MidiTrackingController(
        MidiMappingConfig(
            input_name=args.midi_input_name,
            channel=args.midi_channel,
            toggle_note=args.midi_toggle_note,
            enable_note=args.midi_enable_note,
            disable_note=args.midi_disable_note,
            reacquire_note=args.midi_reacquire_note,
        ),
        start_enabled=args.start_enabled,
    )

    last_ts = time.perf_counter()

    try:
        stream.start()
        midi_controller.connect()
        LOGGER.info("Milestone 5 ready. Press q to exit.")

        while True:
            source_frame = stream.read()
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now

            midi_state = midi_controller.poll()
            if midi_controller.consume_reacquire_request():
                selector.reset()

            if source_frame is None:
                waiting = build_waiting_frame(stream.is_connected, midi_state.tracking_enabled)
                cv2.imshow(args.window_name, waiting)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            virtual_frame = camera.render(source_frame)
            tracks, ran_inference = tracker.maybe_track(virtual_frame)
            frame_height, frame_width = virtual_frame.shape[:2]
            target = selector.select(tracks, frame_width, frame_height) if midi_state.tracking_enabled else None
            pan_velocity, tilt_velocity, error_x, error_y = update_tracking_control(
                camera=camera,
                target=target,
                frame=virtual_frame,
                pan_pid=pan_pid,
                tilt_pid=tilt_pid,
                dt=dt,
                deadzone_x=args.deadzone_x,
                deadzone_y=args.deadzone_y,
                tracking_enabled=midi_state.tracking_enabled,
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
                tracking_enabled=midi_state.tracking_enabled,
                midi_port=args.midi_input_name,
                last_midi_event=midi_state.last_event,
                current_target_id=selector.current_target_id,
                model_path=args.model,
            )

            source_preview = source_frame.copy()
            draw_source_viewport(source_preview, camera)

            cv2.imshow("Milestone 5 - Source", source_preview)
            cv2.imshow(args.window_name, annotated_output)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Milestone 5 MIDI auto tracking failed.")
        return 1
    finally:
        midi_controller.close()
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
    tracking_enabled: bool,
) -> tuple[float, float, float, float]:
    if (not tracking_enabled) or target is None:
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


def draw_target_marker(frame, target: TrackedPerson | None) -> None:
    if target is None:
        return
    x1, y1, x2, y2 = target.bbox_xyxy
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
    cv2.putText(frame, "auto target", (x1, max(20, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


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
    tracking_enabled: bool,
    midi_port: str | None,
    last_midi_event: str,
    current_target_id: int | None,
    model_path: str,
) -> None:
    tracker_stats = tracker.stats
    target_id = "none" if target is None or target.tracker_id is None else str(target.tracker_id)
    selector_target_id = "none" if current_target_id is None else str(current_target_id)
    lines = [
        f"Tracking enabled: {tracking_enabled}",
        f"Target ID: {target_id} | Selector lock: {selector_target_id}",
        f"Model: {model_path}",
        f"Inference runs: {tracker_stats.inference_runs}",
        f"Inference FPS: {tracker_stats.effective_inference_fps:.1f}",
        f"Inference step: {'YOLO' if ran_inference else 'cached'}",
        f"Error X: {error_x:.3f} | Error Y: {error_y:.3f}",
        f"Pan vel: {pan_velocity:.3f} | Tilt vel: {tilt_velocity:.3f}",
        f"MIDI port: {midi_port or 'none'} | Last MIDI: {last_midi_event}",
    ]
    for index, line in enumerate(lines):
        y = 30 + (index * 28)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)


def build_waiting_frame(is_connected: bool, tracking_enabled: bool):
    status = "Opening local video..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, f"Tracking enabled: {tracking_enabled}", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Loading tracker, selector and MIDI...", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
