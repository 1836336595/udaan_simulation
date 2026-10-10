"""Estimate payload motion from sampled position and attitude measurements."""

from __future__ import annotations

import math

import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _hat(vector):
    x, y, z = _vector(vector, "rotation vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=float)


def _project_so3(rotation):
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=float).reshape(3, 3))
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _rotation_exp(vector):
    vector = _vector(vector, "rotation vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-8:
        return np.eye(3) + _hat(vector)
    axis_hat = _hat(vector / angle)
    return (
        np.eye(3)
        + math.sin(angle) * axis_hat
        + (1.0 - math.cos(angle)) * (axis_hat @ axis_hat)
    )


def _rotation_log(rotation):
    rotation = _project_so3(rotation)
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1.0e-7:
        return _vee(0.5 * (rotation - rotation.T))
    sine = math.sin(angle)
    if abs(sine) < 1.0e-6:
        axis = np.sqrt(np.maximum(0.0, (np.diag(rotation) + 1.0) * 0.5))
        index = int(np.argmax(axis))
        if axis[index] < 1.0e-7:
            return np.zeros(3)
        if index == 0:
            axis[1] = rotation[0, 1] / max(2.0 * axis[0], 1.0e-7)
            axis[2] = rotation[0, 2] / max(2.0 * axis[0], 1.0e-7)
        elif index == 1:
            axis[0] = rotation[0, 1] / max(2.0 * axis[1], 1.0e-7)
            axis[2] = rotation[1, 2] / max(2.0 * axis[1], 1.0e-7)
        else:
            axis[0] = rotation[0, 2] / max(2.0 * axis[2], 1.0e-7)
            axis[1] = rotation[1, 2] / max(2.0 * axis[2], 1.0e-7)
        norm = float(np.linalg.norm(axis))
        return angle * (axis / norm if norm > 1.0e-12 else np.array([1.0, 0.0, 0.0]))
    return angle * _vee((rotation - rotation.T) / (2.0 * sine))


class PayloadStateObserver:
    """Causal alpha-beta-gamma and SO(3) observer driven by pose only."""

    def __init__(
        self,
        position_gain=0.35,
        velocity_gain=0.08,
        acceleration_gain=0.02,
        attitude_gain=0.35,
        angular_rate_gain=0.08,
        max_dt=0.05,
        max_velocity_mps=(1.0, 1.0, 1.0),
        max_acceleration_mps2=(8.0, 8.0, 8.0),
        max_body_rate_rps=(8.0, 8.0, 8.0),
        min_samples=3,
    ):
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.acceleration_gain = float(acceleration_gain)
        self.attitude_gain = float(attitude_gain)
        self.angular_rate_gain = float(angular_rate_gain)
        self.max_dt = float(max_dt)
        self.max_velocity = self._limit_vector(max_velocity_mps, "max_velocity_mps")
        self.max_acceleration = self._limit_vector(
            max_acceleration_mps2, "max_acceleration_mps2"
        )
        self.max_body_rate = self._limit_vector(max_body_rate_rps, "max_body_rate_rps")
        self.min_samples = int(min_samples)
        gains = (
            self.position_gain,
            self.velocity_gain,
            self.acceleration_gain,
            self.attitude_gain,
            self.angular_rate_gain,
        )
        if (
            not all(math.isfinite(value) and 0.0 < value <= 1.0 for value in gains)
            or not math.isfinite(self.max_dt)
            or self.max_dt <= 0.0
            or self.min_samples < 2
        ):
            raise ValueError("invalid payload pose observer parameters")
        self.reset()

    @staticmethod
    def _limit_vector(value, name):
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size == 1:
            vector = np.full(3, float(vector[0]))
        if vector.size != 3 or not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError(f"{name} must contain three positive finite values")
        return vector

    def reset(self):
        self._previous_time = None
        self.position = None
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.rotation = np.eye(3)
        self.body_rate = np.zeros(3)
        self.sample_count = 0
        self.derivatives_valid = False

    def _snapshot(self):
        return {
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "body_rate": self.body_rate.copy(),
            "derivatives_valid": bool(self.derivatives_valid),
            "sample_count": int(self.sample_count),
        }

    def update(self, position, rotation, timestamp):
        position = _vector(position, "observer position")
        rotation = _project_so3(rotation)
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            self.reset()
            return None
        if self._previous_time is None or self.position is None:
            self.position = position.copy()
            self.rotation = rotation.copy()
            self._previous_time = timestamp
            self.sample_count = 1
            return self._snapshot()

        dt = timestamp - self._previous_time
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.max_dt:
            self.reset()
            self.position = position.copy()
            self.rotation = rotation.copy()
            self._previous_time = timestamp
            self.sample_count = 1
            return self._snapshot()

        dt2 = dt * dt
        predicted_position = self.position + dt * self.velocity + 0.5 * dt2 * self.acceleration
        predicted_velocity = self.velocity + dt * self.acceleration
        residual = position - predicted_position
        self.position = predicted_position + self.position_gain * residual
        self.velocity = predicted_velocity + self.velocity_gain * residual / dt
        self.acceleration += self.acceleration_gain * residual / dt2
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        self.acceleration = np.clip(
            self.acceleration, -self.max_acceleration, self.max_acceleration
        )

        predicted_rotation = self.rotation @ _rotation_exp(self.body_rate * dt)
        attitude_residual = _rotation_log(predicted_rotation.T @ rotation)
        self.rotation = _project_so3(
            predicted_rotation @ _rotation_exp(self.attitude_gain * attitude_residual)
        )
        self.body_rate += self.angular_rate_gain * attitude_residual / dt
        self.body_rate = np.clip(self.body_rate, -self.max_body_rate, self.max_body_rate)
        self._previous_time = timestamp
        self.sample_count += 1
        self.derivatives_valid = self.sample_count >= self.min_samples
        return self._snapshot()
