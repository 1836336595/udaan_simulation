"""Plot CrazySwarm slung-load simulation CSV diagnostics.

Examples:
    python scripts/plot_crazyswarm_slung_csv.py
    python scripts/plot_crazyswarm_slung_csv.py logs/run.csv
    python scripts/plot_crazyswarm_slung_csv.py logs/run.csv --phase HOVER --save hover.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = REPOSITORY_ROOT / "logs"
VEHICLES = ("CF3", "CF4", "CF5")
PHASES = (
    "GROUND_SLACK",
    "INDEPENDENT_TAKEOFF",
    "TAKEUP",
    "TENSION_RAMP",
    "REFERENCE_LIFT",
    "HOVER",
    "LANDING_TAUT",
    "LANDING_RELEASE",
    "LANDED",
    "EMERGENCY_LANDING",
)
LOAD_FIELDS = (
    "payload_position_x",
    "payload_position_y",
    "payload_position_z",
    "payload_target_x",
    "payload_target_y",
    "payload_target_z",
    "payload_position_error_x",
    "payload_position_error_y",
    "payload_position_error_z",
    "payload_roll_rad",
    "payload_pitch_rad",
    "payload_yaw_rad",
)
OBSERVED_LOAD_FIELDS = (
    "payload_observed_position_x",
    "payload_observed_position_y",
    "payload_observed_position_z",
    "payload_observed_velocity_x",
    "payload_observed_velocity_y",
    "payload_observed_velocity_z",
    "payload_observed_acceleration_x",
    "payload_observed_acceleration_y",
    "payload_observed_acceleration_z",
    "payload_observed_body_rate_x",
    "payload_observed_body_rate_y",
    "payload_observed_body_rate_z",
    "payload_observed_roll_rad",
    "payload_observed_pitch_rad",
    "payload_observed_yaw_rad",
    "payload_observation_derivatives_valid",
    "payload_observed_position_error_x",
    "payload_observed_position_error_y",
    "payload_observed_position_error_z",
    "payload_observed_velocity_error_x",
    "payload_observed_velocity_error_y",
    "payload_observed_velocity_error_z",
)
VEHICLE_FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "cable_tension_n",
    "desired_tension_n",
    "command_thrust_newton",
    "actual_thrust_newton",
    "link_direction_x",
    "link_direction_y",
    "link_direction_z",
    "desired_link_direction_x",
    "desired_link_direction_y",
    "desired_link_direction_z",
)


def _float(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def latest_csv(log_directory=LOG_DIRECTORY):
    candidates = list(Path(log_directory).glob("crazyswarm_slung_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No crazyswarm_slung_*.csv files in {log_directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _thin_indices(count, maximum):
    if maximum <= 0 or count <= maximum:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def load_csv(path, phase=None, max_points=30000):
    """Load one payload sample per tick and separate per-vehicle cable data."""
    path = Path(path).expanduser()
    payload = {name: [] for name in LOAD_FIELDS}
    payload.update({name: [] for name in OBSERVED_LOAD_FIELDS})
    payload["time"] = []
    payload["phase"] = []
    vehicle_data = {
        vehicle: {name: [] for name in (*VEHICLE_FIELDS, "time")}
        for vehicle in VEHICLES
    }
    gains = set()

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"control_time_s", "flight_phase", "vehicle_id", *LOAD_FIELDS, *VEHICLE_FIELDS}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))

        for row in reader:
            stage = row["flight_phase"]
            if phase is not None and stage != phase:
                continue
            vehicle = row["vehicle_id"]
            if vehicle not in vehicle_data:
                continue
            time_s = _float(row, "control_time_s")
            if vehicle == "CF3":
                payload["time"].append(time_s)
                payload["phase"].append(stage)
                for field in LOAD_FIELDS:
                    payload[field].append(_float(row, field))
                for field in OBSERVED_LOAD_FIELDS:
                    payload[field].append(_float(row, field))
            vehicle_data[vehicle]["time"].append(time_s)
            for field in VEHICLE_FIELDS:
                vehicle_data[vehicle][field].append(_float(row, field))
            scale = row.get("transport_link_gain_scale", "")
            if scale:
                gains.add(scale)

    if not payload["time"]:
        suffix = f" for phase {phase}" if phase else ""
        raise ValueError(f"No usable CF3 samples found in {path}{suffix}")

    payload["time"] = np.asarray(payload["time"], dtype=float)
    payload["phase"] = np.asarray(payload["phase"], dtype=str)
    for field in LOAD_FIELDS:
        payload[field] = np.asarray(payload[field], dtype=float)
    for field in OBSERVED_LOAD_FIELDS:
        payload[field] = np.asarray(payload[field], dtype=float)
    payload["payload_yaw_deg"] = np.rad2deg(np.unwrap(payload["payload_yaw_rad"]))
    payload["payload_observed_yaw_deg"] = np.rad2deg(
        np.unwrap(payload["payload_observed_yaw_rad"])
    )

    for vehicle, values in vehicle_data.items():
        values["time"] = np.asarray(values["time"], dtype=float)
        for field in VEHICLE_FIELDS:
            values[field] = np.asarray(values[field], dtype=float)
        actual = np.column_stack(
            [values[f"link_direction_{axis}"] for axis in "xyz"]
        )
        desired = np.column_stack(
            [values[f"desired_link_direction_{axis}"] for axis in "xyz"]
        )
        actual_norm = np.linalg.norm(actual, axis=1)
        desired_norm = np.linalg.norm(desired, axis=1)
        denominator = actual_norm * desired_norm
        dot = np.divide(
            np.einsum("ij,ij->i", actual, desired),
            denominator,
            out=np.full(len(denominator), np.nan),
            where=denominator > 1.0e-12,
        )
        values["link_angle_error_deg"] = np.rad2deg(
            np.arccos(np.clip(dot, -1.0, 1.0))
        )

    indices = _thin_indices(len(payload["time"]), max_points)
    payload = {name: values[indices] for name, values in payload.items()}
    for values in vehicle_data.values():
        vehicle_indices = _thin_indices(len(values["time"]), max_points)
        for name, series in values.items():
            values[name] = series[vehicle_indices]

    return {
        "path": path,
        "payload": payload,
        "vehicles": vehicle_data,
        "gain_scale": ", ".join(sorted(gains)) if gains else "unknown",
    }


def _phase_segments(time_s, phases):
    if len(time_s) == 0:
        return []
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
    segments = []
    start = 0
    for index in range(1, len(phases)):
        if phases[index] != phases[start]:
            segments.append((phases[start], time_s[start], time_s[index]))
            start = index
    segments.append((phases[start], time_s[start], time_s[-1] + dt))
    return segments


def create_figure(data):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })
    payload = data["payload"]
    vehicles = data["vehicles"]
    time_s = payload["time"]
    phases = payload["phase"]
    segments = _phase_segments(time_s, phases)
    phase_order = list(dict.fromkeys(phases.tolist()))
    phase_cmap = plt.get_cmap("tab10", max(1, len(phase_order)))
    phase_colors = {name: phase_cmap(i) for i, name in enumerate(phase_order)}
    vehicle_colors = {"CF3": "#0072B2", "CF4": "#D55E00", "CF5": "#009E73"}
    axis_colors = {"x": "#0072B2", "y": "#D55E00", "z": "#009E73"}

    figure = plt.figure(figsize=(17, 13), constrained_layout=False)
    grid = figure.add_gridspec(
        4,
        2,
        width_ratios=(1.08, 1.0),
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.91,
        hspace=0.50,
        wspace=0.26,
    )
    trajectory = figure.add_subplot(grid[:2, 0], projection="3d")
    position_error = figure.add_subplot(grid[0, 1])
    attitude = figure.add_subplot(grid[1, 1])
    link_angle = figure.add_subplot(grid[2, 0])
    tension = figure.add_subplot(grid[2, 1])
    height = figure.add_subplot(grid[3, 0])
    thrust = figure.add_subplot(grid[3, 1])

    actual_load = np.column_stack([
        payload[f"payload_position_{axis}"] for axis in "xyz"
    ])
    target_load = np.column_stack([
        payload[f"payload_target_{axis}"] for axis in "xyz"
    ])
    observed_load = np.column_stack([
        payload[f"payload_observed_position_{axis}"] for axis in "xyz"
    ])
    trajectory.plot(*actual_load.T, color="#0072B2", linewidth=1.8, label="Load")
    if np.any(np.isfinite(observed_load)):
        trajectory.plot(
            *observed_load.T,
            color="#009E73",
            linewidth=1.2,
            linestyle=":",
            label="Load pose observed by controller",
        )
    trajectory.plot(
        *target_load.T,
        color="#D55E00",
        linewidth=1.3,
        linestyle="--",
        label="Load reference",
    )
    for vehicle, values in vehicles.items():
        position = np.column_stack([values[f"position_{axis}"] for axis in "xyz"])
        trajectory.plot(
            *position.T,
            color=vehicle_colors[vehicle],
            linewidth=0.9,
            alpha=0.8,
            label=vehicle,
        )
        height.plot(
            values["time"],
            values["position_z"],
            color=vehicle_colors[vehicle],
            linewidth=0.9,
            label=vehicle,
        )
    trajectory.scatter(*actual_load[0], color="#0072B2", s=24, marker="o")
    trajectory.scatter(*actual_load[-1], color="#0072B2", s=30, marker="x")
    trajectory.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Height (m)", title="3D trajectories")
    trajectory.legend(loc="upper left", fontsize=8)
    all_points = [actual_load, observed_load, target_load]
    all_points.extend(
        np.column_stack([values[f"position_{axis}"] for axis in "xyz"])
        for values in vehicles.values()
    )
    finite_points = np.vstack(all_points)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    if len(finite_points):
        center = np.mean([finite_points.min(axis=0), finite_points.max(axis=0)], axis=0)
        span = max(float(np.max(np.ptp(finite_points, axis=0))), 0.5)
        half = span * 0.55
        trajectory.set_xlim(center[0] - half, center[0] + half)
        trajectory.set_ylim(center[1] - half, center[1] + half)
        trajectory.set_zlim(center[2] - half, center[2] + half)
        trajectory.set_box_aspect((1, 1, 1))

    for axis in "xyz":
        observed_error = payload[f"payload_observed_position_error_{axis}"]
        if not np.any(np.isfinite(observed_error)):
            observed_error = payload[f"payload_position_error_{axis}"]
        position_error.plot(
            time_s,
            observed_error,
            color=axis_colors[axis],
            linewidth=1.0,
            label=f"e{axis}",
        )
    position_error.set(
        xlabel="Time (s)",
        ylabel="Error (m)",
        title="Load position error (controller observation)",
    )
    position_error.legend(loc="upper right", ncol=3, fontsize=8)
    position_error.grid(True, alpha=0.25)

    for axis, label in (("roll", "Roll"), ("pitch", "Pitch"), ("yaw", "Yaw")):
        observed = payload[f"payload_observed_{axis}_rad"]
        values = np.rad2deg(observed)
        if axis == "yaw":
            values = payload["payload_observed_yaw_deg"]
        if not np.any(np.isfinite(values)):
            values = (
                payload["payload_yaw_deg"]
                if axis == "yaw"
                else np.rad2deg(payload[f"payload_{axis}_rad"])
            )
        attitude.plot(time_s, values, linewidth=1.0, label=label)
    attitude.axhline(0.0, color="#555555", linewidth=0.7, linestyle=":")
    attitude.set(xlabel="Time (s)", ylabel="Angle (deg)", title="Load attitude (yaw unwrapped)")
    attitude.legend(loc="upper right", ncol=3, fontsize=8)
    attitude.grid(True, alpha=0.25)

    for vehicle, values in vehicles.items():
        link_angle.plot(
            values["time"],
            values["link_angle_error_deg"],
            color=vehicle_colors[vehicle],
            linewidth=1.0,
            label=vehicle,
        )
    link_angle.axhline(15.0, color="#555555", linewidth=0.8, linestyle=":", label="15 deg")
    link_angle.set(xlabel="Time (s)", ylabel="Angle (deg)", title="Cable direction error")
    link_angle.legend(loc="upper right", ncol=2, fontsize=8)
    link_angle.grid(True, alpha=0.25)

    for vehicle, values in vehicles.items():
        color = vehicle_colors[vehicle]
        tension.plot(
            values["time"], values["cable_tension_n"],
            color=color, linewidth=1.0, label=f"{vehicle} actual",
        )
        tension.plot(
            values["time"], values["desired_tension_n"],
            color=color, linewidth=0.9, linestyle="--", label=f"{vehicle} desired",
        )
    tension.set(xlabel="Time (s)", ylabel="Tension (N)", title="Cable tension")
    tension.legend(loc="upper right", ncol=2, fontsize=7)
    tension.grid(True, alpha=0.25)

    height.plot(time_s, payload["payload_position_z"], color="#0072B2", linewidth=1.4, label="Load truth")
    if np.any(np.isfinite(payload["payload_observed_position_z"])):
        height.plot(
            time_s,
            payload["payload_observed_position_z"],
            color="#009E73",
            linewidth=1.0,
            linestyle=":",
            label="Observed load pose",
        )
    height.plot(time_s, payload["payload_target_z"], color="#D55E00", linestyle="--", label="Load reference")
    height.axhline(0.0, color="#555555", linewidth=0.7, linestyle=":", label="Ground")
    height.set(xlabel="Time (s)", ylabel="Height (m)", title="Height")
    height.legend(loc="upper right", ncol=2, fontsize=7)
    height.grid(True, alpha=0.25)

    for vehicle, values in vehicles.items():
        color = vehicle_colors[vehicle]
        thrust.plot(
            values["time"], values["command_thrust_newton"],
            color=color, linewidth=0.9, linestyle="--", label=f"{vehicle} command",
        )
        thrust.plot(
            values["time"], values["actual_thrust_newton"],
            color=color, linewidth=1.0, label=f"{vehicle} actual",
        )
    thrust.set(xlabel="Time (s)", ylabel="Thrust (N)", title="Commanded and actual thrust")
    thrust.legend(loc="upper right", ncol=2, fontsize=7)
    thrust.grid(True, alpha=0.25)

    time_axes = (position_error, attitude, link_angle, tension, height, thrust)
    for axes in time_axes:
        for stage, start, stop in segments:
            axes.axvspan(start, stop, color=phase_colors[stage], alpha=0.055, linewidth=0)
        axes.set_xlim(time_s[0], time_s[-1] if len(time_s) == 1 else time_s[-1] + np.median(np.diff(time_s)))

    phases_text = "  >  ".join(phase_order)
    figure.suptitle(
        f"CrazySwarm slung-load CSV diagnostics | link scale={data['gain_scale']}\n"
        f"{data['path'].name}  |  {phases_text}",
        fontsize=13,
    )
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot CrazySwarm slung-load CSV diagnostics.")
    parser.add_argument("csv", nargs="?", type=Path, help="CSV path; defaults to the newest log in logs/")
    parser.add_argument("--phase", choices=PHASES, help="Show only one flight phase.")
    parser.add_argument("--save", type=Path, help="Save a PNG or PDF copy of the figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    parser.add_argument("--max-points", type=int, default=30000, help="Maximum plotted samples per series; 0 keeps all samples.")
    args = parser.parse_args(argv)
    if args.max_points < 0:
        parser.error("--max-points must be zero or positive")

    path = args.csv.expanduser() if args.csv is not None else latest_csv()
    try:
        data = load_csv(path, phase=args.phase, max_points=args.max_points)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = create_figure(data)
    sample_count = len(data["payload"]["time"])
    print(
        f"CSV: {data['path']} | samples: {sample_count} | "
        f"time: {data['payload']['time'][0]:.3f}-{data['payload']['time'][-1]:.3f} s | "
        f"link scale: {data['gain_scale']}"
    )
    if args.save is not None:
        output = args.save.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
