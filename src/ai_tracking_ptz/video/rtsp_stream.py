from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamStats:
    frames_read: int = 0
    frames_dropped: int = 0
    last_frame_ts: float = 0.0


class RTSPVideoStream:
    def __init__(
        self,
        rtsp_url: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        rtsp_transport: str = "tcp",
        reconnect_delay: float = 2.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 3000,
        use_ffmpeg: bool = True,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = reconnect_delay
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.use_ffmpeg = use_ffmpeg

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._last_consumed_frame_id = -1
        self._connected = False
        self._stats = StreamStats()

    @property
    def stats(self) -> StreamStats:
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            LOGGER.warning("RTSPVideoStream is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()

    def read(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None

            frame = self._latest_frame.copy()
            if self._latest_frame_id != self._last_consumed_frame_id:
                dropped_count = max(0, self._latest_frame_id - self._last_consumed_frame_id - 1)
                self._stats.frames_dropped += dropped_count
                self._last_consumed_frame_id = self._latest_frame_id
            return frame

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._open_capture():
                time.sleep(self.reconnect_delay)
                continue

            while not self._stop_event.is_set():
                if self._capture is None:
                    break

                ok, frame = self._capture.read()
                if not ok or frame is None:
                    LOGGER.warning("Frame read failed. Reconnecting RTSP stream.")
                    self._connected = False
                    self._release_capture()
                    time.sleep(self.reconnect_delay)
                    break

                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._stats.frames_read += 1
                    self._stats.last_frame_ts = time.time()

    def _open_capture(self) -> bool:
        self._release_capture()
        backend = cv2.CAP_FFMPEG if self.use_ffmpeg else cv2.CAP_ANY

        try:
            if self.use_ffmpeg and self.rtsp_transport:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.rtsp_transport}"

            capture = cv2.VideoCapture(self.rtsp_url, backend)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)

            if self.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not capture.isOpened():
                LOGGER.error("Unable to open RTSP stream: %s", self.rtsp_url)
                capture.release()
                return False

            self._capture = capture
            self._connected = True
            LOGGER.info("RTSP stream connected successfully.")
            return True
        except Exception:
            LOGGER.exception("Unexpected error opening RTSP stream.")
            self._release_capture()
            return False

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False
