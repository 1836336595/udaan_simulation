"""MuJoCo scene construction for the CrazySwarm slung-load model."""

from __future__ import annotations

import numpy as np

from ...crazyswarm_slung_parameters import default_config


def _vector_text(vector):
    return " ".join(f"{float(value):.12g}" for value in vector)


def initial_vehicle_positions(config):
    points = config.attachment_array
    centroid = np.mean(points[:, :2], axis=0)
    positions = []
    for point in points:
        radial = np.array([point[0] - centroid[0], point[1] - centroid[1], 0.0])
        if np.linalg.norm(radial) < 1.0e-9:
            radial = np.array([1.0, 0.0, 0.0])
        radial /= np.linalg.norm(radial)
        positions.append(
            [
                point[0] + config.takeup_outward_offset * radial[0],
                point[1] + config.takeup_outward_offset * radial[1],
                config.body_size[2] / 2.0,
            ]
        )
    return np.asarray(positions)


def build_mjcf(config=None):
    """Build the scenario MJCF in memory using each aircraft's cable length."""
    config = default_config() if config is None else config
    x, y, z = config.payload_size
    inertia = config.payload_inertia
    body_x, body_y, body_z = config.body_size
    start_positions = initial_vehicle_positions(config)
    payload_pos = np.array([0.0, 0.0, z / 2.0])
    colors = (
        (0.18, 0.50, 0.82, 1.0),
        (0.88, 0.42, 0.18, 1.0),
        (0.28, 0.68, 0.40, 1.0),
    )
    lines = [
        '<mujoco model="crazyswarm_slung_three_cf">',
        '  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>',
        f'  <option timestep="{config.timestep:.9g}" gravity="0 0 -9.81" '
        'integrator="implicitfast"/>',
        '  <size njmax="3000" nconmax="500"/>',
        '  <visual><map znear="0.005" zfar="20"/></visual>',
        '  <default>',
        '    <geom friction="0.8 0.005 0.0001" condim="3" solref="0.005 1" '
        'solimp="0.95 0.99 0.001"/>',
        '  </default>',
        '  <worldbody>',
        '    <light directional="true" diffuse="0.9 0.9 0.9" '
        'specular="0.25 0.25 0.25" pos="0 0 4" dir="0.1 0.1 -1"/>',
        '    <geom name="ground" type="plane" size="4 4 0.1" pos="0 0 0" rgba="0.38 0.42 0.45 1"/>',
        f'    <body name="payload" pos="{_vector_text(payload_pos)}">',
        '      <freejoint name="payload_free"/>',
        f'      <inertial pos="0 0 0" mass="{config.payload_mass:.9g}" '
        f'diaginertia="{_vector_text(inertia)}"/>',
        f'      <geom name="payload_geom" type="box" '
        f'size="{_vector_text(np.array([x, y, z]) / 2.0)}" rgba="0.88 0.78 0.34 1"/>',
    ]
    for index, point in enumerate(config.attachment_points):
        lines.append(
            f'      <site name="payload_attach_{index}" pos="{_vector_text(point)}" '
            'size="0.004" rgba="0.9 0.2 0.15 1"/>'
        )
    lines.append("    </body>")

    for vehicle, position, rgba in zip(config.vehicles, start_positions, colors, strict=True):
        rgba_text = _vector_text(rgba)
        inertia_text = _vector_text(config.vehicle_inertia)
        body_size_text = _vector_text(np.array([body_x, body_y, body_z]) / 2.0)
        arm_offset = config.arm_length / np.sqrt(2.0)
        lines.extend(
            [
                f'    <body name="{vehicle.name}" pos="{_vector_text(position)}">',
                f'      <freejoint name="{vehicle.name}_free"/>',
                f'      <inertial pos="0 0 0" mass="{vehicle.mass:.9g}" '
                f'diaginertia="{inertia_text}"/>',
                f'      <geom name="{vehicle.name}_body" type="box" '
                f'size="{body_size_text}" rgba="{rgba_text}"/>',
                f'      <geom name="{vehicle.name}_arm_a" type="capsule" '
                f'fromto="{-arm_offset:.9g} {-arm_offset:.9g} 0 '
                f'{arm_offset:.9g} {arm_offset:.9g} 0" size="0.002" '
                'contype="0" conaffinity="0" rgba="0.2 0.22 0.24 1"/>',
                f'      <geom name="{vehicle.name}_arm_b" type="capsule" '
                f'fromto="{-arm_offset:.9g} {arm_offset:.9g} 0 '
                f'{arm_offset:.9g} {-arm_offset:.9g} 0" size="0.002" '
                'contype="0" conaffinity="0" rgba="0.2 0.22 0.24 1"/>',
                f'      <site name="{vehicle.name}_cable_anchor" pos="0 0 0" '
                'size="0.003" rgba="0.15 0.15 0.15 1"/>',
            ]
        )
        rotor_xy = config.arm_length / np.sqrt(2.0)
        rotor_positions = (
            (rotor_xy, rotor_xy),
            (-rotor_xy, rotor_xy),
            (-rotor_xy, -rotor_xy),
            (rotor_xy, -rotor_xy),
        )
        for rotor_index, (rotor_x, rotor_y) in enumerate(rotor_positions):
            lines.append(
                f'      <geom name="{vehicle.name}_rotor_{rotor_index}" type="cylinder" '
                f'pos="{rotor_x:.9g} {rotor_y:.9g} 0.002" '
                f'size="{config.rotor_radius:.9g} 0.001" contype="0" conaffinity="0" '
                'rgba="0.72 0.76 0.78 0.65"/>'
            )
            guard_center_radius = config.prop_guard_outer_diameter / 2.0 - config.prop_guard_bar_radius
            for segment in range(config.prop_guard_segments):
                angle_start = 2.0 * np.pi * segment / config.prop_guard_segments
                angle_end = 2.0 * np.pi * (segment + 1) / config.prop_guard_segments
                start = (
                    rotor_x + guard_center_radius * np.cos(angle_start),
                    rotor_y + guard_center_radius * np.sin(angle_start),
                    config.prop_guard_z_offset,
                )
                end = (
                    rotor_x + guard_center_radius * np.cos(angle_end),
                    rotor_y + guard_center_radius * np.sin(angle_end),
                    config.prop_guard_z_offset,
                )
                lines.append(
                    f'      <geom name="{vehicle.name}_prop_guard_{rotor_index}_{segment}" '
                    'type="capsule" '
                    f'fromto="{_vector_text((*start, *end))}" '
                    f'size="{config.prop_guard_bar_radius:.9g}" '
                    'contype="0" conaffinity="0" rgba="0.16 0.18 0.20 0.95"/>'
                )
        lines.append("    </body>")

    lines.append("  </worldbody>")
    lines.append("  <tendon>")
    for index, (vehicle, length) in enumerate(
        zip(config.vehicles, config.cable_lengths, strict=True)
    ):
        lines.extend(
            [
                f'    <spatial name="cable_{vehicle.name}" limited="true" '
                f'range="0 {length:.9g}" width="0.002" rgba="0.9 0.9 0.9 1" '
                'damping="0">',
                f'      <site site="{vehicle.name}_cable_anchor"/>',
                f'      <site site="payload_attach_{index}"/>',
                "    </spatial>",
            ]
        )
    lines.extend(["  </tendon>", "</mujoco>"])
    return "\n".join(lines)
