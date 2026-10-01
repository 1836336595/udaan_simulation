"""CrazySwarm CTBR and cooperative rigid-payload controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..crazyswarm_slung_parameters import CrazyflieParameters as CrazyflieParameters

_Z_FLIP = np.diag([1.0, 1.0, -1.0])


def z_up_to_down(vector):
    """Convert a polar vector from MuJoCo z-up to MATLAB z-down coordinates."""
    return _Z_FLIP @ np.asarray(vector, dtype=float)


def z_down_to_up(vector):
    """Convert a polar vector from MATLAB z-down to MuJoCo z-up coordinates."""
    return _Z_FLIP @ np.asarray(vector, dtype=float)


def _axial_down_to_up(vector):
    """Convert an axial vector between frames related by a reflection."""
    return -_Z_FLIP @ np.asarray(vector, dtype=float)


def _hat(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def _normalize(vector, fallback):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.asarray(fallback, dtype=float).copy()
    return vector / norm


def _project_so3(rotation):
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=float))
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


class CrazySwarmSlungController:
    """Stateful controller ported from CrazySwarm's MATLAB slung-load model.

    The cooperative force-allocation and link equations are evaluated in the
    MATLAB z-down convention. Returned CTBR commands use MuJoCo body axes.
    """

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.independent_integrals = np.zeros((3, 3))
        self.payload_position_integral = np.zeros(3)
        self.link_integrals = np.zeros((3, 3))

    @staticmethod
    def _rotation_down(rotation_up):
        return _Z_FLIP @ np.asarray(rotation_up, dtype=float) @ _Z_FLIP

    @staticmethod
    def _body_rate_down(body_rate_up):
        return _axial_down_to_up(body_rate_up)

    def independent_commands(self, vehicle_states, desired_positions, desired_velocities, dt):
        """Compute per-aircraft MATLAB independent CTBR commands."""
        positions = np.asarray(desired_positions, dtype=float).reshape(3, 3)
        velocities = np.asarray(desired_velocities, dtype=float).reshape(3, 3)
        commands = []
        e3 = np.array([0.0, 0.0, 1.0])
        for index, (state, params) in enumerate(zip(vehicle_states, self.config.vehicles, strict=True)):
            position = z_up_to_down(state["position"])
            velocity = z_up_to_down(state["velocity"])
            desired_position = z_up_to_down(positions[index])
            desired_velocity = z_up_to_down(velocities[index])
            error_position = position - desired_position
            error_velocity = velocity - desired_velocity
            if np.linalg.norm(error_position) < params.independent_integral_gate:
                integral_rate = error_velocity + 0.5 * error_position
                self.independent_integrals[:, index] += float(dt) * integral_rate
                self.independent_integrals[:, index] = np.clip(
                    self.independent_integrals[:, index],
                    -np.asarray(params.independent_integral_limit),
                    np.asarray(params.independent_integral_limit),
                )

            feedback_acceleration = (
                -np.asarray(params.independent_position_gain) * error_position
                - np.asarray(params.independent_velocity_gain) * error_velocity
                - np.asarray(params.independent_integral_gain)
                * self.independent_integrals[:, index]
            )
            feedback_acceleration = np.clip(
                feedback_acceleration,
                -np.asarray(params.independent_max_feedback_acceleration),
                np.asarray(params.independent_max_feedback_acceleration),
            )
            desired_acceleration_down = np.zeros(3)
            force_down = params.mass * (
                feedback_acceleration + desired_acceleration_down - 9.81 * e3
            )
            commands.append(
                self._force_to_command(
                    state,
                    params,
                    force_down,
                    params.independent_attitude_gain,
                    (0.0, 0.0, 0.0),
                    minimum_thrust=params.min_flight_thrust,
                )
            )
        return commands

    def cooperative_commands(
        self,
        payload_state,
        vehicle_states,
        desired,
        dt,
        link_gain_scale=0.0,
    ):
        """Compute MATLAB's allocated payload and cable-direction commands."""
        if len(vehicle_states) != 3:
            raise ValueError("CrazySwarm slung-load control requires exactly three vehicles")
        dt = float(np.clip(dt, 0.001, 0.05))
        e3 = np.array([0.0, 0.0, 1.0])
        gravity = 9.81
        payload_mass = self.config.payload_mass
        payload_size = np.asarray(self.config.payload_size, dtype=float)
        inertia = np.diag(
            [
                payload_mass * (payload_size[1] ** 2 + payload_size[2] ** 2) / 12.0,
                payload_mass * (payload_size[0] ** 2 + payload_size[2] ** 2) / 12.0,
                payload_mass * (payload_size[0] ** 2 + payload_size[1] ** 2) / 12.0,
            ]
        )

        p0 = z_up_to_down(payload_state["position"])
        v0 = z_up_to_down(payload_state["velocity"])
        r0 = self._rotation_down(payload_state["rotation"])
        omega0 = self._body_rate_down(payload_state["body_rate"])
        pd = z_up_to_down(desired["position"])
        vd = z_up_to_down(desired.get("velocity", np.zeros(3)))
        ad = z_up_to_down(desired.get("acceleration", np.zeros(3)))
        rd = self._rotation_down(desired.get("rotation", np.eye(3)))
        omegad = self._body_rate_down(desired.get("body_rate", np.zeros(3)))
        omegadd = self._body_rate_down(desired.get("body_rate_dot", np.zeros(3)))

        error_position = p0 - pd
        error_velocity = v0 - vd
        self.payload_position_integral += dt * (
            error_velocity + self.config.payload_integral_c1 * error_position
        )
        self.payload_position_integral = np.clip(
            self.payload_position_integral,
            -np.asarray(self.config.payload_integral_limit),
            np.asarray(self.config.payload_integral_limit),
        )
        desired_load_acceleration = (
            ad
            - np.asarray(self.config.payload_position_gain) * error_position
            - np.asarray(self.config.payload_velocity_gain) * error_velocity
            - np.asarray(self.config.payload_integral_gain) * self.payload_position_integral
        )
        desired_force = payload_mass * (desired_load_acceleration - gravity * e3)

        error_rotation = 0.5 * _vee(rd.T @ r0 - r0.T @ rd)
        error_omega = omega0 - r0.T @ rd @ omegad
        bandwidth = 2.0 * np.pi * np.asarray(self.config.payload_attitude_bandwidth_hz)
        attitude_gain = np.diag(inertia) * bandwidth**2
        rate_gain = 2.0 * self.config.payload_attitude_damping * bandwidth * np.diag(inertia)
        feedforward_rate = r0.T @ rd @ omegad
        desired_moment = (
            -attitude_gain * error_rotation
            - rate_gain * error_omega
            + _hat(feedforward_rate) @ inertia @ feedforward_rate
            + inertia @ r0.T @ rd @ omegadd
        )
        if not self.config.payload_yaw_enabled:
            desired_moment[2] = 0.0
        omega0_dot = np.linalg.solve(
            inertia, desired_moment - np.cross(omega0, inertia @ omega0)
        )

        allocation = np.zeros((6, 9))
        attachment_down = [z_up_to_down(row) for row in self.config.attachment_points]
        for index, rho in enumerate(attachment_down):
            allocation[:3, 3 * index : 3 * index + 3] = np.eye(3)
            allocation[3:, 3 * index : 3 * index + 3] = _hat(rho)
        gram = allocation @ allocation.T
        if np.linalg.cond(gram) > 1.0 / self.config.tension_pinv_tolerance:
            raise ValueError("CrazySwarm payload tension allocation is singular")
        rhs = np.concatenate([r0.T @ desired_force, desired_moment])
        mu_body = allocation.T @ np.linalg.solve(gram, rhs)
        mu_body += self._internal_tension_bias(allocation, attachment_down)
        mu_desired = r0 @ mu_body.reshape(3, 3).T

        commands = []
        link_units = np.zeros((3, 3))
        desired_link_units = np.zeros((3, 3))
        for index, (state, params, rho) in enumerate(
            zip(vehicle_states, self.config.vehicles, attachment_down, strict=True)
        ):
            anchor = z_up_to_down(state["anchor_position"])
            anchor_velocity = z_up_to_down(state["anchor_velocity"])
            attachment = p0 + r0 @ rho
            relative = attachment - anchor
            q = _normalize(relative, [0.0, 0.0, 1.0])
            payload_omega_world = r0 @ omega0
            rho_world = r0 @ rho
            attachment_velocity = v0 + np.cross(payload_omega_world, rho_world)
            relative_velocity = attachment_velocity - anchor_velocity
            qdot = (np.eye(3) - np.outer(q, q)) @ relative_velocity / max(
                float(np.linalg.norm(relative)), 1.0e-6
            )
            omega_link = np.cross(q, qdot)

            mu_id = mu_desired[:, index]
            mu_i = q * float(q @ mu_id)
            qid = _normalize(-mu_id, [0.0, 0.0, 1.0])
            eq = np.cross(qid, q)
            self.link_integrals[:, index] = np.clip(
                self.link_integrals[:, index] + dt * eq,
                -np.asarray(self.config.link_integral_limit),
                np.asarray(self.config.link_integral_limit),
            )
            gain_scale = float(np.clip(link_gain_scale, 0.0, 1.0))
            correction = (
                -gain_scale * self.config.link_kq * eq
                - gain_scale * self.config.link_komega * omega_link
                - gain_scale * self.config.link_integral_gain * self.link_integrals[:, index]
            )
            q_hat_sq = _hat(q) @ _hat(q)
            ai = (
                desired_load_acceleration
                - gravity * e3
                + r0 @ (_hat(omega0) @ (_hat(omega0) @ rho))
                - r0 @ _hat(rho) @ omega0_dot
            )
            vehicle_mass = params.mass
            cable_length = float(self.config.cable_lengths[index])
            u_parallel = (
                mu_i
                + vehicle_mass * cable_length * float(omega_link @ omega_link) * q
                + vehicle_mass * q * float(q @ ai)
            )
            u_perpendicular = (
                vehicle_mass * cable_length * np.cross(q, correction)
                - vehicle_mass * q_hat_sq @ ai
            )
            force_down = u_parallel + u_perpendicular
            commands.append(
                self._force_to_command(
                    state,
                    params,
                    force_down,
                    params.transport_attitude_gain,
                    params.transport_rate_gain,
                    minimum_thrust=params.min_flight_thrust,
                )
            )
            link_units[:, index] = q
            desired_link_units[:, index] = qid

        return {
            "commands": commands,
            "position_error": error_position,
            "position_error_world": z_down_to_up(error_position),
            "velocity_error": error_velocity,
            "velocity_error_world": z_down_to_up(error_velocity),
            "attitude_error": error_rotation,
            "attitude_error_world": _axial_down_to_up(error_rotation),
            "desired_force": desired_force,
            "desired_force_world": z_down_to_up(desired_force),
            "desired_moment": desired_moment,
            "desired_moment_world": _axial_down_to_up(desired_moment),
            "link_units": link_units,
            "desired_link_units": desired_link_units,
            "desired_tensions": np.linalg.norm(mu_desired, axis=0),
        }

    def _internal_tension_bias(self, allocation, attachments):
        desired = np.zeros((3, 3))
        for index, rho in enumerate(attachments):
            radial = np.array([rho[0], rho[1], 0.0])
            radial = _normalize(
                radial,
                [np.cos(2.0 * np.pi * index / 3.0), np.sin(2.0 * np.pi * index / 3.0), 0.0],
            )
            desired[:, index] = (
                self.config.outward_bias_fraction
                * self.config.payload_mass
                * 9.81
                / np.sqrt(3.0)
                * radial
            )
        _, singular, vh = np.linalg.svd(allocation)
        rank = int(np.sum(singular > self.config.tension_pinv_tolerance))
        null_basis = vh[rank:, :].T
        bias = null_basis @ (null_basis.T @ desired.reshape(-1, order="F"))
        blocks = bias.reshape(3, 3)
        rms = float(np.sqrt(np.mean(np.sum(blocks * blocks, axis=0))))
        target_rms = min(
            self.config.outward_bias_fraction * self.config.payload_mass * 9.81 / np.sqrt(3.0),
            self.config.outward_bias_max_n,
        )
        if rms > 1.0e-12:
            bias *= target_rms / rms
        return bias

    @staticmethod
    def _force_to_command(state, params, force_down, attitude_gain, rate_gain, minimum_thrust):
        force_down = np.asarray(force_down, dtype=float)
        rotation_down = CrazySwarmSlungController._rotation_down(state["rotation"])
        body_rate_down = CrazySwarmSlungController._body_rate_down(state["body_rate"])
        norm_force = float(np.linalg.norm(force_down))
        if norm_force < 1.0e-9:
            desired_b3 = rotation_down[:, 2]
        else:
            desired_b3 = -force_down / norm_force

        max_tilt = float(params.max_tilt_rad)
        tilt = float(np.arccos(np.clip(desired_b3[2], -1.0, 1.0)))
        if tilt > max_tilt:
            horizontal = _normalize(desired_b3[:2], [1.0, 0.0])
            desired_b3 = np.array(
                [np.sin(max_tilt) * horizontal[0], np.sin(max_tilt) * horizontal[1], np.cos(max_tilt)]
            )

        heading = np.array([1.0, 0.0, 0.0])
        b1 = (np.eye(3) - np.outer(desired_b3, desired_b3)) @ heading
        if np.linalg.norm(b1) < 1.0e-6:
            b1 = (np.eye(3) - np.outer(desired_b3, desired_b3)) @ np.array([0.0, 1.0, 0.0])
        b1 = _normalize(b1, [1.0, 0.0, 0.0])
        b2 = _normalize(np.cross(desired_b3, b1), [0.0, 1.0, 0.0])
        b1 = _normalize(np.cross(b2, desired_b3), [1.0, 0.0, 0.0])
        desired_rotation_down = _project_so3(np.column_stack([b1, b2, desired_b3]))

        error_rotation = 0.5 * _vee(
            desired_rotation_down.T @ rotation_down - rotation_down.T @ desired_rotation_down
        )
        error_norm = float(np.linalg.norm(error_rotation))
        if error_norm > 2.0:
            error_rotation *= 2.0 / error_norm
        omega_command_down = (
            -np.asarray(attitude_gain) * error_rotation
            - np.asarray(rate_gain) * body_rate_down
        )
        body_rate_limit = np.asarray(params.max_body_rate)
        omega_command_down = np.clip(omega_command_down, -body_rate_limit, body_rate_limit)
        body_rate_command_up = _axial_down_to_up(omega_command_down)

        thrust = -float(force_down @ rotation_down[:, 2])
        thrust = float(np.clip(thrust, minimum_thrust, params.max_command_thrust))
        return VehicleCommand(
            thrust=thrust,
            body_rate=body_rate_command_up,
            desired_force_world=z_down_to_up(force_down),
        )


@dataclass
class VehicleCommand:
    thrust: float
    body_rate: np.ndarray
    desired_force_world: np.ndarray | None = None
