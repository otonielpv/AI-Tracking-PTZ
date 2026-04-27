from __future__ import annotations

import logging
from dataclasses import dataclass

import mido


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MidiMappingConfig:
    input_name: str | None = None
    channel: int = 0
    toggle_note: int = 60
    enable_note: int = 61
    disable_note: int = 62
    reacquire_note: int = 63


@dataclass(slots=True)
class MidiControllerState:
    tracking_enabled: bool = True
    reacquire_requested: bool = False
    last_event: str = "startup"


class MidiTrackingController:
    def __init__(self, config: MidiMappingConfig, start_enabled: bool = True) -> None:
        self.config = config
        self.state = MidiControllerState(tracking_enabled=start_enabled, last_event="startup")
        self._input = None

    @property
    def is_connected(self) -> bool:
        return self._input is not None

    @staticmethod
    def list_inputs() -> list[str]:
        return list(mido.get_input_names())

    def connect(self) -> None:
        if not self.config.input_name:
            LOGGER.info("No MIDI input configured. Tracking state will stay local to the app.")
            return
        self._input = mido.open_input(self.config.input_name)
        LOGGER.info("Connected to MIDI input: %s", self.config.input_name)

    def close(self) -> None:
        if self._input is not None:
            self._input.close()
            self._input = None

    def poll(self) -> MidiControllerState:
        if self._input is None:
            return self.state

        for message in self._input.iter_pending():
            self._handle_message(message)
        return self.state

    def consume_reacquire_request(self) -> bool:
        if not self.state.reacquire_requested:
            return False
        self.state.reacquire_requested = False
        return True

    def _handle_message(self, message) -> None:
        if message.type != "note_on" or getattr(message, "velocity", 0) <= 0:
            return
        if getattr(message, "channel", -1) != self.config.channel:
            return

        note = getattr(message, "note", None)
        if note == self.config.toggle_note:
            self.state.tracking_enabled = not self.state.tracking_enabled
            self.state.last_event = f"toggle:{note}"
        elif note == self.config.enable_note:
            self.state.tracking_enabled = True
            self.state.last_event = f"enable:{note}"
        elif note == self.config.disable_note:
            self.state.tracking_enabled = False
            self.state.last_event = f"disable:{note}"
        elif note == self.config.reacquire_note:
            self.state.reacquire_requested = True
            self.state.last_event = f"reacquire:{note}"
