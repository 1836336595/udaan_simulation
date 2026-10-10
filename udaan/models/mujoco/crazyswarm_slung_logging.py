"""CSV diagnostics for the CrazySwarm slung-load simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _csv_vector_fields(prefix):
    return [f"{prefix}_{axis}" for axis in ("x", "y", "z")]


def rotation_euler_xyz(rotation):
    rotation = np.asarray(rotation, dtype=float)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw])


CRAZYSWARM_CSV_FIELDS = [
    "control_time_s",
    "phase_time_s",
    "flight_phase",
    "control_mode",
    "cooperative_blend",
    "vehicle_id",
    *_csv_vector_fields("position"),
    *_csv_vector_fields("velocity"),
    *_csv_vector_fields("acceleration"),
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    *_csv_vector_fields("body_rate"),
    *_csv_vector_fields("vehicle_target"),
    *_csv_vector_fields("vehicle_target_velocity"),
    *_csv_vector_fields("position_error"),
    *_csv_vector_fields("velocity_error"),
    "command_thrust_newton",
    "actual_thrust_newton",
    *_csv_vector_fields("command_rate"),
    *_csv_vector_fields("desired_force"),
    *_csv_vector_fields("actual_force"),
    *_csv_vector_fields("payload_position"),
    *_csv_vector_fields("payload_velocity"),
    *_csv_vector_fields("payload_acceleration"),
    "payload_roll_rad",
    "payload_pitch_rad",
    "payload_yaw_rad",
    *_csv_vector_fields("payload_body_rate"),
    *_csv_vector_fields("payload_observed_position"),
    *_csv_vector_fields("payload_observed_velocity"),
    *_csv_vector_fields("payload_observed_acceleration"),
    "payload_observed_roll_rad",
    "payload_observed_pitch_rad",
    "payload_observed_yaw_rad",
    *_csv_vector_fields("payload_observed_body_rate"),
    "payload_observation_derivatives_valid",
    "payload_observation_sample_count",
    "payload_observation_sample_time_s",
    "payload_pose_sample_rate_hz",
    "payload_state_estimator",
    "payload_velocity_filter_cutoff_hz",
    "payload_observer_position_gain",
    "payload_observer_velocity_gain",
    "payload_observer_acceleration_gain",
    "payload_observer_attitude_gain",
    "payload_observer_angular_rate_gain",
    "payload_observer_max_dt_s",
    *_csv_vector_fields("payload_observer_max_velocity"),
    *_csv_vector_fields("payload_observer_max_acceleration"),
    *_csv_vector_fields("payload_observer_max_body_rate"),
    *_csv_vector_fields("payload_target"),
    *_csv_vector_fields("payload_target_velocity"),
    *_csv_vector_fields("payload_target_acceleration"),
    *_csv_vector_fields("payload_position_error"),
    *_csv_vector_fields("payload_velocity_error"),
    *_csv_vector_fields("payload_observed_position_error"),
    *_csv_vector_fields("payload_observed_velocity_error"),
    *_csv_vector_fields("payload_attitude_error"),
    *_csv_vector_fields("payload_desired_force"),
    *_csv_vector_fields("payload_desired_moment"),
    *_csv_vector_fields("payload_position_integral"),
    "cable_length_m",
    "cable_length_error_m",
    "cable_velocity_mps",
    "cable_tension_n",
    "desired_tension_n",
    *_csv_vector_fields("link_direction"),
    *_csv_vector_fields("desired_link_direction"),
    *_csv_vector_fields("link_direction_error"),
    "vehicle_mass_kg",
    *_csv_vector_fields("vehicle_kp"),
    *_csv_vector_fields("vehicle_kd"),
    *_csv_vector_fields("vehicle_ki"),
    *_csv_vector_fields("vehicle_integral_limit"),
    *_csv_vector_fields("independent_kp"),
    *_csv_vector_fields("independent_kd"),
    *_csv_vector_fields("independent_ki"),
    *_csv_vector_fields("independent_integral_limit"),
    *_csv_vector_fields("independent_max_feedback_acceleration"),
    *_csv_vector_fields("independent_attitude_gain"),
    *_csv_vector_fields("transport_attitude_gain"),
    *_csv_vector_fields("transport_rate_gain"),
    *_csv_vector_fields("max_body_rate"),
    "vehicle_max_total_thrust_n",
    "vehicle_max_command_thrust_n",
    "vehicle_min_flight_thrust_n",
    "vehicle_max_tilt_rad",
    "independent_integral_gate_m",
    "payload_mass_kg",
    *_csv_vector_fields("payload_kp"),
    *_csv_vector_fields("payload_kd"),
    *_csv_vector_fields("payload_ki"),
    *_csv_vector_fields("payload_integral_limit"),
    *_csv_vector_fields("payload_attitude_bandwidth_hz"),
    "payload_attitude_damping",
    "payload_integral_c1",
    "payload_yaw_enabled",
    "link_kq",
    "link_komega",
    "link_ki",
    *_csv_vector_fields("link_integral_limit"),
    "transport_link_gain_scale",
    "prop_guard_outer_diameter_m",
    "prop_guard_bar_diameter_m",
    "prop_guard_z_offset_m",
    "prop_guard_segments",
    "ramp_position_hold_gain_scale",
    "outward_bias_fraction",
    "outward_bias_max_n",
    "tension_pinv_tolerance",
    "thrust_time_constant_s",
    "rate_loop_bandwidth_x",
    "rate_loop_bandwidth_y",
    "rate_loop_bandwidth_z",
]


class CrazySwarmSlungCsvLogger:
    """Write stable, per-vehicle control and slung-load diagnostics."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CRAZYSWARM_CSV_FIELDS)
        self._writer.writeheader()
        self._writes_since_flush = 0
        self.flush()

    def write(self, row):
        if self._file.closed:
            return
        self._writer.writerow(row)
        self._writes_since_flush += 1
        if self._writes_since_flush >= 25:
            self.flush()

    def flush(self):
        if not self._file.closed:
            self._file.flush()
            self._writes_since_flush = 0

    def close(self):
        if not self._file.closed:
            self.flush()
            self._file.close()
