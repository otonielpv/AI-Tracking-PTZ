from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_min: float = -1.0
    integral_max: float = 1.0


@dataclass(slots=True)
class PIDState:
    integral: float = 0.0
    previous_error: float = 0.0
    has_previous: bool = False


class PIDController:
    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self.state = PIDState()

    def reset(self) -> None:
        self.state = PIDState()

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            proportional = self.config.kp * error
            return self._clamp_output(proportional)

        self.state.integral += error * dt
        self.state.integral = self._clamp(
            self.state.integral,
            self.config.integral_min,
            self.config.integral_max,
        )

        derivative = 0.0
        if self.state.has_previous:
            derivative = (error - self.state.previous_error) / dt

        output = (
            (self.config.kp * error)
            + (self.config.ki * self.state.integral)
            + (self.config.kd * derivative)
        )

        self.state.previous_error = error
        self.state.has_previous = True
        return self._clamp_output(output)

    def _clamp_output(self, value: float) -> float:
        return self._clamp(value, self.config.output_min, self.config.output_max)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
