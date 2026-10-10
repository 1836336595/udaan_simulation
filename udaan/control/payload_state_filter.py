"""Estimate payload derivatives from pose samples using second-order filters."""

from __future__ import annotations

import math

import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _rotation_vector(rotation):
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    skew = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1.0e-7:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) >= 1.0e-6:
        return angle * skew / (2.0 * sine)
    axis = np.sqrt(np.maximum(0.0, (np.diag(rotation) + 1.0) * 0.5))
    index = int(np.argmax(axis))
    if axis[index] < 1.0e-7:
        return np.zeros(3)
    if index == 0:
        axis[1] = rotation[0, 1] / (2.0 * axis[0])
        axis[2] = rotation[0, 2] / (2.0 * axis[0])
    elif index == 1:
        axis[0] = rotation[0, 1] / (2.0 * axis[1])
        axis[2] = rotation[1, 2] / (2.0 * axis[1])
    else:
        axis[0] = rotation[0, 2] / (2.0 * axis[2])
        axis[1] = rotation[1, 2] / (2.0 * axis[2])
    norm = float(np.linalg.norm(axis))
    return angle * axis / norm if norm > 1.0e-12 else np.zeros(3)


class _SecondOrderButterworth:
    def __init__(self, cutoff_hz, max_dt):
        self.cutoff_hz = float(cutoff_hz)
        self.max_dt = float(max_dt)
        if (
            not math.isfinite(self.cutoff_hz)
            or self.cutoff_hz <= 0.0
            or not math.isfinite(self.max_dt)
            or self.max_dt <= 0.0
            or 2.0 * self.cutoff_hz * self.max_dt > 0.8
        ):
            raise ValueError("invalid second-order pose filter parameters")
        self.reset()

    def reset(self):
        self._previous_time = None
        self._previous_input = None
        self._previous_input_2 = None
        self._previous_output = None
        self._previous_output_2 = None

    def update(self, value, timestamp):
        value = _vector(value, "filter input")
        if self._previous_time is None:
            self._previous_time = float(timestamp)
            self._previous_input = value.copy()
            self._previous_input_2 = value.copy()
            self._previous_output = value.copy()
            self._previous_output_2 = value.copy()
            return value.copy(), np.zeros(3)

        dt = float(timestamp) - self._previous_time
        normalized_cutoff = 2.0 * self.cutoff_hz * dt
        omega = math.pi * normalized_cutoff
        cosine = math.cos(omega)
        alpha = math.sin(omega) / (2.0 * math.sqrt(0.5))
        a0 = 1.0 + alpha
        b0 = (1.0 - cosine) / (2.0 * a0)
        b1 = (1.0 - cosine) / a0
        b2 = b0
        a1 = -2.0 * cosine / a0
        a2 = (1.0 - alpha) / a0
        filtered = (
            b0 * value
            + b1 * self._previous_input
            + b2 * self._previous_input_2
            - a1 * self._previous_output
            - a2 * self._previous_output_2
        )
        derivative = (filtered - self._previous_output) / dt
        self._previous_input_2 = self._previous_input
        self._previous_input = value.copy()
        self._previous_output_2 = self._previous_output
        self._previous_output = filtered.copy()
        self._previous_time = float(timestamp)
        return filtered, derivative


class _SecondOrderButterworthPosition:
    """Butterworth-smoothed position with velocity/acceleration filter outputs."""

    def __init__(self, cutoff_hz, max_dt):
        self.cutoff_hz = float(cutoff_hz)
        self.max_dt = float(max_dt)
        if (
            not math.isfinite(self.cutoff_hz)
            or self.cutoff_hz <= 0.0
            or not math.isfinite(self.max_dt)
            or self.max_dt <= 0.0
            or 2.0 * self.cutoff_hz * self.max_dt > 0.8
        ):
            raise ValueError("invalid second-order position filter parameters")
        self.reset()

    def reset(self):
        self._previous_time = None
        self._previous_input = None
        self._previous_input_2 = None
        self._position = None
        self._position_2 = None
        self._velocity = None
        self._velocity_2 = None
        self._acceleration = None
        self._acceleration_2 = None

    def _seed(self, position, timestamp):
        self._previous_time = float(timestamp)
        self._previous_input = position.copy()
        self._previous_input_2 = position.copy()
        self._position = position.copy()
        self._position_2 = position.copy()
        self._velocity = np.zeros(3)
        self._velocity_2 = np.zeros(3)
        self._acceleration = np.zeros(3)
        self._acceleration_2 = np.zeros(3)
        return position.copy(), np.zeros(3), np.zeros(3)

    @staticmethod
    def _filter(value, previous_input, previous_input_2,
                previous_output, previous_output_2, numerator, a1, a2):
        return (
            numerator[0] * value
            + numerator[1] * previous_input
            + numerator[2] * previous_input_2
            - a1 * previous_output
            - a2 * previous_output_2
        )

    def update(self, position, timestamp):
        position = _vector(position, "position filter input")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            self.reset()
            return None, None, None
        if self._previous_time is None:
            return self._seed(position, timestamp)

        dt = timestamp - self._previous_time
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.max_dt:
            self.reset()
            return self._seed(position, timestamp)

        omega_n = 2.0 * math.pi * self.cutoff_hz
        bilinear_scale = 2.0 / dt
        scale2 = bilinear_scale * bilinear_scale
        omega2 = omega_n * omega_n
        denominator0 = scale2 + math.sqrt(2.0) * omega_n * bilinear_scale + omega2
        a1 = (-2.0 * scale2 + 2.0 * omega2) / denominator0
        a2 = (scale2 - math.sqrt(2.0) * omega_n * bilinear_scale + omega2) / denominator0
        position_numerator = omega2 / denominator0 * np.array([1.0, 2.0, 1.0])
        velocity_numerator = omega2 * bilinear_scale / denominator0 * np.array(
            [1.0, 0.0, -1.0]
        )
        acceleration_numerator = omega2 * scale2 / denominator0 * np.array(
            [1.0, -2.0, 1.0]
        )

        filtered_position = self._filter(
            position, self._previous_input, self._previous_input_2,
            self._position, self._position_2, position_numerator, a1, a2,
        )
        velocity = self._filter(
            position, self._previous_input, self._previous_input_2,
            self._velocity, self._velocity_2, velocity_numerator, a1, a2,
        )
        acceleration = self._filter(
            position, self._previous_input, self._previous_input_2,
            self._acceleration, self._acceleration_2, acceleration_numerator, a1, a2,
        )

        self._previous_input_2 = self._previous_input
        self._previous_input = position.copy()
        self._position_2, self._position = self._position, filtered_position
        self._velocity_2, self._velocity = self._velocity, velocity
        self._acceleration_2, self._acceleration = self._acceleration, acceleration
        self._previous_time = timestamp
        return filtered_position, velocity, acceleration


class PayloadPoseLowPassFilter:
    """Filter position before deriving linear kinematics; filter attitude rates."""

    def __init__(self, cutoff_hz=3.0, max_dt=0.05, min_samples=3):
        self.cutoff_hz = float(cutoff_hz)
        self.max_dt = float(max_dt)
        self.min_samples = int(min_samples)
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        self._position_filter = _SecondOrderButterworthPosition(cutoff_hz, max_dt)
        self._angular_rate_filter = _SecondOrderButterworth(cutoff_hz, max_dt)
        self.reset()

    def reset(self):
        self._previous_time = None
        self._previous_rotation = None
        self._position_filter.reset()
        self._angular_rate_filter.reset()
        self.position = None
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.sample_count = 0
        self.derivatives_valid = False

    def _snapshot(self):
        return {
            "position": None if self.position is None else self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "body_rate": self.body_rate.copy(),
            "derivatives_valid": bool(self.derivatives_valid),
            "sample_count": int(self.sample_count),
        }

    def update(self, position, rotation, timestamp):
        position = _vector(position, "filter position")
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        timestamp = float(timestamp)
        if not np.all(np.isfinite(rotation)) or not math.isfinite(timestamp):
            self.reset()
            return None

        if self._previous_time is None:
            self._previous_time = timestamp
            self._previous_rotation = rotation.copy()
            self.position = position.copy()
            self.sample_count = 1
            return self._snapshot()

        dt = timestamp - self._previous_time
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.max_dt:
            self.reset()
            self._previous_time = timestamp
            self._previous_rotation = rotation.copy()
            self.position = position.copy()
            self.sample_count = 1
            return self._snapshot()

        relative_rotation = self._previous_rotation.T @ rotation
        raw_body_rate = _rotation_vector(relative_rotation) / dt
        self.position, self.velocity, self.acceleration = self._position_filter.update(
            position, timestamp
        )
        self.body_rate, _ = self._angular_rate_filter.update(raw_body_rate, timestamp)
        self._previous_time = timestamp
        self._previous_rotation = rotation.copy()
        self.sample_count += 1
        self.derivatives_valid = self.sample_count >= self.min_samples
        return self._snapshot()
