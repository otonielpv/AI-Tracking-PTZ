from __future__ import annotations

import time
from dataclasses import dataclass

from ai_tracking_ptz.tracking.yolo_person_tracker import TrackedPerson


@dataclass(slots=True)
class TargetSelectorConfig:
    area_weight: float = 0.45
    center_weight: float = 0.25
    confidence_weight: float = 0.10
    persistence_weight: float = 0.20
    current_target_bonus: float = 0.20
    max_persistence_seconds: float = 3.0
    target_hold_seconds: float = 1.0


@dataclass(slots=True)
class TrackMemory:
    sightings: int = 0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0


class AutoTargetSelector:
    def __init__(self, config: TargetSelectorConfig | None = None) -> None:
        self.config = config or TargetSelectorConfig()
        self._track_memory: dict[int, TrackMemory] = {}
        self._current_target_id: int | None = None
        self._target_acquired_ts: float = 0.0

    @property
    def current_target_id(self) -> int | None:
        return self._current_target_id

    def reset(self) -> None:
        self._current_target_id = None
        self._target_acquired_ts = 0.0

    def select(self, tracks: list[TrackedPerson], frame_width: int, frame_height: int) -> TrackedPerson | None:
        now = time.time()
        self._update_memory(tracks, now)
        self._prune_memory(now)

        if not tracks:
            self.reset()
            return None

        scored_tracks = [
            (self._score_track(track, frame_width, frame_height, now), track)
            for track in tracks
        ]
        scored_tracks.sort(key=lambda item: item[0], reverse=True)

        best_score, best_track = scored_tracks[0]
        current_track = self._find_track_by_id(tracks, self._current_target_id)
        if current_track is not None:
            current_score = self._score_track(current_track, frame_width, frame_height, now)
            if (now - self._target_acquired_ts) < self.config.target_hold_seconds and current_score >= (best_score - 0.15):
                return current_track

        self._current_target_id = best_track.tracker_id
        self._target_acquired_ts = now
        return best_track

    def _update_memory(self, tracks: list[TrackedPerson], now: float) -> None:
        for track in tracks:
            if track.tracker_id is None:
                continue
            memory = self._track_memory.get(track.tracker_id)
            if memory is None:
                memory = TrackMemory(sightings=0, first_seen_ts=now, last_seen_ts=now)
                self._track_memory[track.tracker_id] = memory
            memory.sightings += 1
            memory.last_seen_ts = now

    def _prune_memory(self, now: float) -> None:
        stale_ids = [
            track_id
            for track_id, memory in self._track_memory.items()
            if (now - memory.last_seen_ts) > self.config.max_persistence_seconds
        ]
        for track_id in stale_ids:
            self._track_memory.pop(track_id, None)
        if self._current_target_id is not None and self._current_target_id not in self._track_memory:
            self.reset()

    def _score_track(self, track: TrackedPerson, frame_width: int, frame_height: int, now: float) -> float:
        x1, y1, x2, y2 = track.bbox_xyxy
        area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, frame_width * frame_height)
        area_score = min(1.0, area / frame_area * 8.0)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        normalized_dx = abs(center_x - (frame_width / 2.0)) / (frame_width / 2.0)
        normalized_dy = abs(center_y - (frame_height / 2.0)) / (frame_height / 2.0)
        center_score = max(0.0, 1.0 - ((normalized_dx + normalized_dy) / 2.0))

        persistence_score = 0.0
        if track.tracker_id is not None and track.tracker_id in self._track_memory:
            memory = self._track_memory[track.tracker_id]
            age = max(0.0, now - memory.first_seen_ts)
            persistence_score = min(1.0, age / self.config.max_persistence_seconds)

        score = (
            (area_score * self.config.area_weight)
            + (center_score * self.config.center_weight)
            + (track.confidence * self.config.confidence_weight)
            + (persistence_score * self.config.persistence_weight)
        )
        if track.tracker_id is not None and track.tracker_id == self._current_target_id:
            score += self.config.current_target_bonus
        return score

    @staticmethod
    def _find_track_by_id(tracks: list[TrackedPerson], tracker_id: int | None) -> TrackedPerson | None:
        if tracker_id is None:
            return None
        for track in tracks:
            if track.tracker_id == tracker_id:
                return track
        return None
