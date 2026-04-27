from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Protocol

import cv2
import mido
from ultralytics import YOLO

from ai_tracking_ptz.camera.onvif_ptz import DEFAULT_ONVIF_PORT, OnvifConnectionConfig, OnvifPTZCamera
from ai_tracking_ptz.camera.virtual_ptz import VirtualPTZCamera, VirtualPTZConfig
from ai_tracking_ptz.control.pid import PIDConfig, PIDController
from ai_tracking_ptz.logging_utils import configure_logging
from ai_tracking_ptz.midi.controller import MidiMappingConfig, MidiTrackingController
from ai_tracking_ptz.tracking.target_selector import AutoTargetSelector, TargetSelectorConfig
from ai_tracking_ptz.tracking.yolo_person_tracker import TrackedPerson, YoloPersonTracker, draw_tracks
from ai_tracking_ptz.video.file_stream import FileVideoStream
from ai_tracking_ptz.video.rtsp_stream import RTSPVideoStream


LOGGER = logging.getLogger(__name__)


class VideoSource(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def stats(self): ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self): ...


class PTZController(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def continuous_move(self, pan_velocity: float = 0.0, tilt_velocity: float = 0.0, zoom_velocity: float = 0.0) -> None: ...

    def stop(self) -> None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for AI Tracking PTZ")
    subparsers = parser.add_subparsers(dest="command", required=True)

    view_parser = subparsers.add_parser("view", help="Open and display a video source")
    add_source_args(view_parser, require_source=True)
    view_parser.add_argument("--window-name", default="AI Tracking PTZ - View", help="OpenCV window name")
    view_parser.add_argument("--log-level", default="INFO", help="Logging level")
    view_parser.set_defaults(handler=handle_view)

    track_parser = subparsers.add_parser("track", help="Run YOLO person tracking on a video source")
    add_source_args(track_parser, require_source=True)
    add_tracker_args(track_parser)
    track_parser.add_argument("--window-name", default="AI Tracking PTZ - Track", help="OpenCV window name")
    track_parser.add_argument("--log-level", default="INFO", help="Logging level")
    track_parser.set_defaults(handler=handle_track)

    ptz_parser = subparsers.add_parser("ptz-test", help="Manual PTZ test for virtual or ONVIF backends")
    ptz_parser.add_argument("--ptz-backend", choices=["virtual", "onvif"], default="virtual", help="PTZ backend")
    add_source_args(ptz_parser, require_source=False)
    add_virtual_ptz_args(ptz_parser)
    add_onvif_args(ptz_parser)
    ptz_parser.add_argument("--pan-speed", type=float, default=0.5, help="Pan velocity 0.0-1.0")
    ptz_parser.add_argument("--tilt-speed", type=float, default=0.5, help="Tilt velocity 0.0-1.0")
    ptz_parser.add_argument("--zoom-speed", type=float, default=0.3, help="Zoom velocity 0.0-1.0")
    ptz_parser.add_argument("--log-level", default="INFO", help="Logging level")
    ptz_parser.set_defaults(handler=handle_ptz_test)

    auto_parser = subparsers.add_parser("auto-track", help="Run automatic AI tracking with PID and optional MIDI")
    add_source_args(auto_parser, require_source=True)
    auto_parser.add_argument("--ptz-backend", choices=["virtual", "onvif"], default="virtual", help="PTZ backend")
    add_virtual_ptz_args(auto_parser)
    add_onvif_args(auto_parser)
    add_tracker_args(auto_parser)
    add_pid_args(auto_parser)
    add_selector_args(auto_parser)
    add_midi_args(auto_parser)
    auto_parser.add_argument("--start-enabled", action="store_true", help="Start with tracking enabled")
    auto_parser.add_argument("--window-name", default="AI Tracking PTZ - Auto Track", help="OpenCV output window name")
    auto_parser.add_argument("--log-level", default="INFO", help="Logging level")
    auto_parser.set_defaults(handler=handle_auto_track)

    export_parser = subparsers.add_parser("export-engine", help="Export a YOLO model to TensorRT")
    export_parser.add_argument("--model", default="yolov8n.pt", help="Path to the source .pt model")
    export_parser.add_argument("--imgsz", type=int, default=640, help="Inference/export image size")
    export_parser.add_argument("--device", default="cuda:0", help="Export device")
    export_parser.add_argument("--half", action="store_true", help="Enable FP16 export when supported")
    export_parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size in GB")
    export_parser.add_argument("--log-level", default="INFO", help="Logging level")
    export_parser.set_defaults(handler=handle_export_engine)

    midi_parser = subparsers.add_parser("list-midi", help="List available MIDI input ports")
    midi_parser.add_argument("--log-level", default="INFO", help="Logging level")
    midi_parser.set_defaults(handler=handle_list_midi)

    return parser


def add_source_args(parser: argparse.ArgumentParser, require_source: bool) -> None:
    source_group = parser.add_mutually_exclusive_group(required=require_source)
    source_group.add_argument("--rtsp-url", help="RTSP URL for the PTZ camera or test stream")
    source_group.add_argument("--video-file", help="Local video file for offline testing")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height")
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp", help="RTSP transport for FFmpeg backend")
    parser.add_argument("--loop-video", action="store_true", help="Loop local video file when it reaches the end")


def add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path. Can be .pt or .engine")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config name")
    parser.add_argument("--imgsz", type=int, default=640, help="Internal inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-inference-fps", type=float, default=15.0, help="Inference rate limit")
    parser.add_argument("--device", default=None, help="Inference device. Example: cpu or cuda:0")


def add_virtual_ptz_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--virtual-width", type=int, default=1280, help="Virtual camera output width")
    parser.add_argument("--virtual-height", type=int, default=720, help="Virtual camera output height")
    parser.add_argument("--virtual-max-zoom", type=float, default=4.0, help="Maximum virtual zoom")


def add_onvif_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None, help="Camera IP or hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_ONVIF_PORT, help="ONVIF service port")
    parser.add_argument("--username", default=None, help="ONVIF username")
    parser.add_argument("--password", default=None, help="ONVIF password")
    parser.add_argument("--wsdl-dir", default=None, help="Optional path to ONVIF WSDL directory")


def add_pid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pid-pan-kp", type=float, default=1.2, help="PID Kp for pan")
    parser.add_argument("--pid-pan-ki", type=float, default=0.0, help="PID Ki for pan")
    parser.add_argument("--pid-pan-kd", type=float, default=0.25, help="PID Kd for pan")
    parser.add_argument("--pid-tilt-kp", type=float, default=1.0, help="PID Kp for tilt")
    parser.add_argument("--pid-tilt-ki", type=float, default=0.0, help="PID Ki for tilt")
    parser.add_argument("--pid-tilt-kd", type=float, default=0.2, help="PID Kd for tilt")
    parser.add_argument("--deadzone-x", type=float, default=0.05, help="Horizontal deadzone as normalized half-width")
    parser.add_argument("--deadzone-y", type=float, default=0.05, help="Vertical deadzone as normalized half-height")


def add_selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selector-area-weight", type=float, default=0.45, help="Auto-target area weight")
    parser.add_argument("--selector-center-weight", type=float, default=0.25, help="Auto-target center weight")
    parser.add_argument("--selector-confidence-weight", type=float, default=0.10, help="Auto-target confidence weight")
    parser.add_argument("--selector-persistence-weight", type=float, default=0.20, help="Auto-target persistence weight")
    parser.add_argument("--selector-target-bonus", type=float, default=0.20, help="Auto-target lock bonus")


def add_midi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--midi-input-name", default=None, help="MIDI input port name")
    parser.add_argument("--midi-channel", type=int, default=0, help="MIDI channel 0-15")
    parser.add_argument("--midi-toggle-note", type=int, default=60, help="MIDI note for toggle tracking")
    parser.add_argument("--midi-enable-note", type=int, default=61, help="MIDI note for enable tracking")
    parser.add_argument("--midi-disable-note", type=int, default=62, help="MIDI note for disable tracking")
    parser.add_argument("--midi-reacquire-note", type=int, default=63, help="MIDI note for reacquire target")


def build_stream(args: argparse.Namespace) -> VideoSource:
    if args.video_file:
        return FileVideoStream(file_path=args.video_file, loop=args.loop_video)
    return RTSPVideoStream(
        rtsp_url=args.rtsp_url,
        width=args.width,
        height=args.height,
        rtsp_transport=args.rtsp_transport,
    )


def build_tracker(args: argparse.Namespace) -> YoloPersonTracker:
    return YoloPersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_fps=args.max_inference_fps,
    )


def build_virtual_ptz(args: argparse.Namespace) -> VirtualPTZCamera:
    return VirtualPTZCamera(
        VirtualPTZConfig(
            output_width=args.virtual_width,
            output_height=args.virtual_height,
            max_zoom=args.virtual_max_zoom,
        )
    )


def build_onvif_ptz(args: argparse.Namespace) -> OnvifPTZCamera:
    if not args.host or not args.username or not args.password:
        raise ValueError("ONVIF backend requires --host, --username and --password.")
    return OnvifPTZCamera(
        OnvifConnectionConfig(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            wsdl_dir=args.wsdl_dir,
        )
    )


def handle_view(args: argparse.Namespace) -> int:
    stream = build_stream(args)
    return run_view_loop(stream, args.window_name)


def handle_track(args: argparse.Namespace) -> int:
    stream = build_stream(args)
    tracker = build_tracker(args)
    return run_tracking_loop(stream, tracker, args.window_name)


def handle_ptz_test(args: argparse.Namespace) -> int:
    if args.ptz_backend == "virtual":
        if not args.video_file:
            raise ValueError("Virtual PTZ test requires --video-file.")
        stream = build_stream(args)
        camera = build_virtual_ptz(args)
        return run_virtual_ptz_test(stream, camera, args.pan_speed, args.tilt_speed, args.zoom_speed)

    camera = build_onvif_ptz(args)
    return run_onvif_ptz_test(camera, args.pan_speed, args.tilt_speed, args.zoom_speed)


def handle_auto_track(args: argparse.Namespace) -> int:
    stream = build_stream(args)
    tracker = build_tracker(args)
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

    if args.ptz_backend == "virtual":
        ptz = build_virtual_ptz(args)
        return run_auto_tracking_loop(
            stream=stream,
            tracker=tracker,
            ptz=ptz,
            pan_pid=pan_pid,
            tilt_pid=tilt_pid,
            selector=selector,
            midi_controller=midi_controller,
            window_name=args.window_name,
            model_path=args.model,
            deadzone_x=args.deadzone_x,
            deadzone_y=args.deadzone_y,
            virtual_mode=True,
        )

    ptz = build_onvif_ptz(args)
    return run_auto_tracking_loop(
        stream=stream,
        tracker=tracker,
        ptz=ptz,
        pan_pid=pan_pid,
        tilt_pid=tilt_pid,
        selector=selector,
        midi_controller=midi_controller,
        window_name=args.window_name,
        model_path=args.model,
        deadzone_x=args.deadzone_x,
        deadzone_y=args.deadzone_y,
        virtual_mode=False,
    )


def handle_export_engine(args: argparse.Namespace) -> int:
    model = YOLO(args.model)
    output_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        workspace=args.workspace,
    )
    LOGGER.info("TensorRT export completed: %s", output_path)
    return 0


def handle_list_midi(args: argparse.Namespace) -> int:
    input_names = list(mido.get_input_names())
    if not input_names:
        print("No MIDI input ports detected.")
        return 0
    for name in input_names:
        print(name)
    return 0


def run_view_loop(stream: VideoSource, window_name: str) -> int:
    try:
        stream.start()
        LOGGER.info("Press q to exit.")
        while True:
            frame = stream.read()
            key = cv2.waitKey(1) & 0xFF
            if frame is None:
                cv2.imshow(window_name, build_waiting_frame(stream.is_connected, "Waiting for source..."))
                if key in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            draw_stream_info(frame, stream)
            cv2.imshow(window_name, frame)
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def run_tracking_loop(stream: VideoSource, tracker: YoloPersonTracker, window_name: str) -> int:
    try:
        stream.start()
        LOGGER.info("Press q to exit.")
        while True:
            frame = stream.read()
            key = cv2.waitKey(1) & 0xFF
            if frame is None:
                cv2.imshow(window_name, build_waiting_frame(stream.is_connected, "Loading tracker..."))
                if key in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            tracks, ran_inference = tracker.maybe_track(frame)
            output = frame.copy()
            draw_tracks(output, tracks)
            draw_tracking_info(output, tracker, len(tracks), ran_inference)
            cv2.imshow(window_name, output)
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def run_virtual_ptz_test(stream: VideoSource, camera: VirtualPTZCamera, pan_speed: float, tilt_speed: float, zoom_speed: float) -> int:
    last_ts = time.perf_counter()
    try:
        stream.start()
        LOGGER.info("Virtual PTZ ready. Use W/A/S/D for tilt/pan, Z/X for zoom, Space to stop, Q to quit.")
        while True:
            frame = stream.read()
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now
            key = cv2.waitKey(1) & 0xFF

            if frame is None:
                cv2.imshow("AI Tracking PTZ - Virtual PTZ", build_waiting_frame(stream.is_connected, "Opening local source..."))
                if key in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            handle_ptz_key(camera, key, pan_speed, tilt_speed, zoom_speed)
            camera.update(dt)

            virtual_frame = camera.render(frame)
            source_preview = frame.copy()
            draw_virtual_viewport(source_preview, camera)
            cv2.imshow("AI Tracking PTZ - Source", source_preview)
            cv2.imshow("AI Tracking PTZ - Virtual PTZ", virtual_frame)
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def run_onvif_ptz_test(camera: OnvifPTZCamera, pan_speed: float, tilt_speed: float, zoom_speed: float) -> int:
    try:
        camera.connect()
        LOGGER.info("ONVIF PTZ ready. Use W/A/S/D for tilt/pan, Z/X for zoom, Space to stop, Q to quit.")
        while True:
            frame = build_ptz_test_frame(camera.is_connected, pan_speed, tilt_speed, zoom_speed, "ONVIF PTZ Test")
            cv2.imshow("AI Tracking PTZ - ONVIF", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                break
            handle_ptz_key(camera, key, pan_speed, tilt_speed, zoom_speed)
        return 0
    finally:
        try:
            camera.stop()
        finally:
            camera.close()
            cv2.destroyAllWindows()


def run_auto_tracking_loop(
    stream: VideoSource,
    tracker: YoloPersonTracker,
    ptz: PTZController,
    pan_pid: PIDController,
    tilt_pid: PIDController,
    selector: AutoTargetSelector,
    midi_controller: MidiTrackingController,
    window_name: str,
    model_path: str,
    deadzone_x: float,
    deadzone_y: float,
    virtual_mode: bool,
) -> int:
    last_ts = time.perf_counter()
    previous_enabled = midi_controller.state.tracking_enabled

    try:
        stream.start()
        if hasattr(ptz, "connect"):
            getattr(ptz, "connect")()
        midi_controller.connect()
        LOGGER.info("Auto tracking ready. Press q to exit.")

        while True:
            source_frame = stream.read()
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now
            key = cv2.waitKey(1) & 0xFF

            midi_state = midi_controller.poll()
            if not midi_state.tracking_enabled and previous_enabled:
                selector.reset()
            previous_enabled = midi_state.tracking_enabled
            if midi_controller.consume_reacquire_request():
                selector.reset()

            if source_frame is None:
                cv2.imshow(window_name, build_waiting_frame(stream.is_connected, "Loading tracking pipeline..."))
                if key in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue

            tracking_frame = source_frame
            if virtual_mode and isinstance(ptz, VirtualPTZCamera):
                tracking_frame = ptz.render(source_frame)

            tracks, ran_inference = tracker.maybe_track(tracking_frame)
            frame_height, frame_width = tracking_frame.shape[:2]
            target = selector.select(tracks, frame_width, frame_height) if midi_state.tracking_enabled else None

            pan_velocity, tilt_velocity, error_x, error_y = update_tracking_control(
                ptz=ptz,
                target=target,
                frame=tracking_frame,
                pan_pid=pan_pid,
                tilt_pid=tilt_pid,
                dt=dt,
                deadzone_x=deadzone_x,
                deadzone_y=deadzone_y,
                tracking_enabled=midi_state.tracking_enabled,
            )

            if virtual_mode and isinstance(ptz, VirtualPTZCamera):
                ptz.update(dt)

            annotated = tracking_frame.copy()
            draw_tracks(annotated, tracks)
            draw_target_marker(annotated, target)
            draw_deadzone(annotated, deadzone_x, deadzone_y)
            draw_auto_track_info(
                annotated,
                tracker=tracker,
                target=target,
                pan_velocity=pan_velocity,
                tilt_velocity=tilt_velocity,
                error_x=error_x,
                error_y=error_y,
                ran_inference=ran_inference,
                tracking_enabled=midi_state.tracking_enabled,
                midi_port=midi_controller.config.input_name,
                last_midi_event=midi_state.last_event,
                current_target_id=selector.current_target_id,
                model_path=model_path,
            )

            if virtual_mode and isinstance(ptz, VirtualPTZCamera):
                source_preview = source_frame.copy()
                draw_virtual_viewport(source_preview, ptz)
                cv2.imshow("AI Tracking PTZ - Source", source_preview)

            cv2.imshow(window_name, annotated)
            if key in (ord("q"), 27):
                break

        return 0
    finally:
        try:
            ptz.stop()
        except Exception:
            LOGGER.debug("Ignoring PTZ stop error during cleanup.", exc_info=True)
        if hasattr(ptz, "close"):
            try:
                getattr(ptz, "close")()
            except Exception:
                LOGGER.debug("Ignoring PTZ close error during cleanup.", exc_info=True)
        midi_controller.close()
        stream.stop()
        cv2.destroyAllWindows()


def update_tracking_control(
    ptz: PTZController,
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
        ptz.stop()
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
    ptz.continuous_move(pan_velocity=pan_velocity, tilt_velocity=tilt_velocity, zoom_velocity=0.0)
    return pan_velocity, tilt_velocity, error_x, error_y


def handle_ptz_key(camera, key: int, pan_speed: float, tilt_speed: float, zoom_speed: float) -> None:
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


def draw_stream_info(frame, stream: VideoSource) -> None:
    stats = stream.stats
    lines = [
        f"Connected: {stream.is_connected}",
        f"Frames read: {stats.frames_read}",
        f"Frames dropped: {stats.frames_dropped}",
    ]
    draw_lines(frame, lines)


def draw_tracking_info(frame, tracker: YoloPersonTracker, active_tracks: int, ran_inference: bool) -> None:
    tracker_stats = tracker.stats
    lines = [
        f"Tracks: {active_tracks}",
        f"Inference runs: {tracker_stats.inference_runs}",
        f"Inference FPS: {tracker_stats.effective_inference_fps:.1f}",
        f"Inference step: {'YOLO' if ran_inference else 'cached'}",
    ]
    draw_lines(frame, lines)


def draw_auto_track_info(
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
    draw_lines(frame, lines)


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
    cv2.rectangle(frame, (center_x - half_width, center_y - half_height), (center_x + half_width, center_y + half_height), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)


def draw_virtual_viewport(frame, camera: VirtualPTZCamera) -> None:
    x1, y1, x2, y2 = camera.describe_viewport(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(frame, "Virtual PTZ viewport", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)


def draw_lines(frame, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        y = 30 + (index * 28)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)


def build_waiting_frame(is_connected: bool, message: str):
    status = "Connecting source..." if not is_connected else "Waiting for frames..."
    frame = cv2.UMat(480, 854, cv2.CV_8UC3).get()
    frame[:] = (20, 20, 20)
    cv2.putText(frame, status, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, message, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to exit", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def build_ptz_test_frame(is_connected: bool, pan_speed: float, tilt_speed: float, zoom_speed: float, title: str):
    frame = cv2.UMat(540, 960, cv2.CV_8UC3).get()
    frame[:] = (16, 16, 16)
    lines = [
        title,
        f"Connected: {is_connected}",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception as exc:
        LOGGER.exception("Command failed.")
        if isinstance(exc, ValueError):
            parser.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
