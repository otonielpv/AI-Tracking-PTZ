from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ai_tracking_ptz.video.rtsp_stream import StreamStats


LOGGER = logging.getLogger(__name__)


class FileVideoStream:
    def __init__(
        self,
        file_path: str,
        loop: bool = True,
        realtime: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.loop = loop
        self.realtime = realtime

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._last_consumed_frame_id = -1
        self._connected = False
        self._stats = StreamStats()
        self._frame_interval_s = 0.0

    @property
    def stats(self) -> StreamStats:
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            LOGGER.warning("FileVideoStream is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="file-reader", daemon=True)
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
        if not self.file_path.exists():
            LOGGER.error("Video file does not exist: %s", self.file_path)
            return

        if not self._open_capture():
            return

        while not self._stop_event.is_set():
            if self._capture is None:
                break

            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self.loop:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                LOGGER.info("Reached end of video file: %s", self.file_path)
                self._connected = False
                break

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._stats.frames_read += 1
                self._stats.last_frame_ts = time.time()

            if self.realtime and self._frame_interval_s > 0:
                time.sleep(self._frame_interval_s)

        self._release_capture()

    def _open_capture(self) -> bool:
        self._release_capture()
        capture = cv2.VideoCapture(str(self.file_path))
        if not capture.isOpened():
            LOGGER.error("Unable to open video file: %s", self.file_path)
            capture.release()
            return False

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self._frame_interval_s = 1.0 / fps
        else:
            self._frame_interval_s = 0.0

        self._capture = capture
        self._connected = True
        LOGGER.info("Video file opened successfully: %s", self.file_path)
        return True

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False
