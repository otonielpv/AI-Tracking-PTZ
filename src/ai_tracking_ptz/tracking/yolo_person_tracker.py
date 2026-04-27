from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackerStats:
    inference_runs: int = 0
    last_inference_ts: float = 0.0
    effective_inference_fps: float = 0.0


@dataclass(slots=True)
class TrackedPerson:
    tracker_id: Optional[int]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class YoloPersonTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_config: str = "botsort.yaml",
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        max_fps: float = 15.0,
        use_half: bool = True,
    ) -> None:
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_fps = max_fps
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and self.device.startswith("cuda")
        self._min_inference_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_inference_monotonic = 0.0
        self._last_tracks: list[TrackedPerson] = []
        self._stats = TrackerStats()

        LOGGER.info(
            "Loading YOLO model=%s device=%s imgsz=%s max_fps=%s half=%s tracker=%s",
            self.model_path,
            self.device,
            self.imgsz,
            self.max_fps,
            self.use_half,
            self.tracker_config,
        )
        self._model = YOLO(self.model_path)

    @property
    def stats(self) -> TrackerStats:
        return self._stats

    def maybe_track(self, frame: np.ndarray) -> tuple[list[TrackedPerson], bool]:
        now = time.monotonic()
        if self._min_inference_interval and (now - self._last_inference_monotonic) < self._min_inference_interval:
            return self._last_tracks, False

        try:
            results = self._model.track(
                source=frame,
                persist=True,
                classes=[0],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_config,
                verbose=False,
                device=self.device,
                half=self.use_half,
            )
        except Exception:
            LOGGER.exception("YOLO tracking inference failed.")
            return self._last_tracks, False

        self._last_inference_monotonic = now
        self._stats.inference_runs += 1

        previous_inference_ts = self._stats.last_inference_ts
        current_inference_ts = time.time()
        self._stats.last_inference_ts = current_inference_ts
        if previous_inference_ts > 0:
            delta = current_inference_ts - previous_inference_ts
            if delta > 0:
                self._stats.effective_inference_fps = 1.0 / delta

        if not results:
            self._last_tracks = []
            return self._last_tracks, True

        self._last_tracks = self._parse_result(results[0])
        return self._last_tracks, True

    def _parse_result(self, result) -> list[TrackedPerson]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy_values = boxes.xyxy.int().cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        id_values = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy_values)

        tracks: list[TrackedPerson] = []
        for index, bbox in enumerate(xyxy_values):
            confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
            tracker_id = id_values[index] if index < len(id_values) else None
            tracks.append(
                TrackedPerson(
                    tracker_id=tracker_id,
                    confidence=confidence,
                    bbox_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                )
            )
        return tracks


def draw_tracks(frame: np.ndarray, tracks: list[TrackedPerson]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox_xyxy
        label_id = "?" if track.tracker_id is None else str(track.tracker_id)
        label = f"person {label_id} {track.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
