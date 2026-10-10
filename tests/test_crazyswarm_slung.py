"""Tests for the CrazySwarm three-vehicle slung-load scenario."""

import csv
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from udaan.models.mujoco import _GlfwViewer
from udaan.control.crazyswarm_slung import (
    CrazySwarmSlungController,
    z_down_to_up,
    z_up_to_down,
)
from udaan.control.payload_state_observer import PayloadStateObserver
from udaan.control.payload_state_filter import PayloadPoseLowPassFilter
from udaan.models.mujoco.crazyswarm_slung import (
    CrazySwarmSlungModel,
    CrazySwarmSlungPhase,
    default_config,
    quintic_profile,
)


class CrazySwarmSlungTests(unittest.TestCase):
    def test_second_order_pose_filter_estimates_translation_and_rotation_rates(self):
        estimator = PayloadPoseLowPassFilter(cutoff_hz=3.0)
        observed = None
        for sample in range(201):
            timestamp = sample / 100.0
            position = np.array([0.2 * timestamp, -0.1 * timestamp, 0.0])
            yaw = 0.3 * timestamp
            rotation = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            observed = estimator.update(position, rotation, timestamp)

        self.assertTrue(observed["derivatives_valid"])
        self.assertGreater(np.linalg.norm(observed["position"] - position), 1e-3)
        np.testing.assert_allclose(observed["position"], position, atol=0.02)
        np.testing.assert_allclose(observed["velocity"], [0.2, -0.1, 0.0], atol=1e-6)
        np.testing.assert_allclose(observed["body_rate"], [0.0, 0.0, 0.3], atol=1e-5)
        np.testing.assert_allclose(observed["acceleration"], 0.0, atol=1e-5)

    def test_payload_estimator_mode_is_configurable(self):
        default = default_config()
        self.assertEqual(default.payload_state_estimator, "observer")
        self.assertEqual(default.payload_velocity_filter_cutoff_hz, 3.0)
        observer_config = replace(default, payload_state_estimator="observer")
        model = CrazySwarmSlungModel(render=False, config=observer_config)
        try:
            self.assertEqual(model.config.payload_state_estimator, "observer")
        finally:
            model.close()

    def test_mujoco_truth_mode_uses_full_payload_state_at_each_step(self):
        config = replace(default_config(), payload_state_estimator="mujoco_truth")
        model = CrazySwarmSlungModel(render=False, config=config)
        try:
            expected_velocity = np.array([0.2, -0.1, 0.3])
            expected_acceleration = np.array([1.0, 2.0, -3.0])
            expected_body_rate = np.array([0.4, -0.5, 0.6])
            model.data.qvel[model._load_dof : model._load_dof + 6] = [
                *expected_velocity, *expected_body_rate
            ]
            model.data.qacc[model._load_dof : model._load_dof + 6] = [
                *expected_acceleration, 0.0, 0.0, 0.0
            ]
            model.data.time = 0.001
            state = model._update_payload_controller_state()

            np.testing.assert_allclose(state["velocity"], expected_velocity)
            np.testing.assert_allclose(state["acceleration"], expected_acceleration)
            np.testing.assert_allclose(state["body_rate"], expected_body_rate)
            self.assertTrue(state["derivatives_valid"])
        finally:
            model.close()

    def test_payload_observer_estimates_derivatives_from_pose_samples(self):
        observer = PayloadStateObserver()
        observed = None
        for sample in range(101):
            timestamp = sample / 100.0
            position = np.array([0.2 * timestamp, 0.0, 0.0])
            yaw = 0.3 * timestamp
            rotation = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            observed = observer.update(position, rotation, timestamp)

        self.assertTrue(observed["derivatives_valid"])
        self.assertGreater(observed["velocity"][0], 0.1)
        self.assertGreater(observed["body_rate"][2], 0.1)
        self.assertTrue(np.all(np.isfinite(observed["acceleration"])))

    def test_payload_control_state_does_not_read_mujoco_derivatives(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            model.data.qvel[model._load_dof : model._load_dof + 6] = [
                0.8, -0.4, 0.3, 0.2, -0.1, 0.9
            ]
            model.data.qacc[model._load_dof : model._load_dof + 6] = [
                2.0, -2.0, 3.0, 1.0, 1.0, -1.0
            ]
            model.data.time = model._payload_pose_sample_period
            state = model._update_payload_controller_state()

            np.testing.assert_allclose(state["position"], model.payload_state()["position"])
            np.testing.assert_allclose(state["rotation"], model.payload_state()["rotation"])
            np.testing.assert_allclose(state["velocity"], 0.0)
            np.testing.assert_allclose(state["acceleration"], 0.0)
            np.testing.assert_allclose(state["body_rate"], 0.0)
            self.assertNotEqual(state["velocity"][0], model.payload_state()["velocity"][0])
        finally:
            model.close()

    def test_viewer_mouse_callbacks_use_supported_camera_api(self):
        viewer = _GlfwViewer.__new__(_GlfwViewer)
        viewer._model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="target"><freejoint/>'
            '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
        )
        viewer._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(viewer._camera)
        viewer._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer._camera.trackbodyid = 1
        viewer._glfw = SimpleNamespace(
            get_window_size=lambda _window: (1200, 900),
            get_cursor_pos=lambda _window: (0.0, 0.0),
        )
        for button in ("left", "right", "middle"):
            viewer._button_left = button == "left"
            viewer._button_right = button == "right"
            viewer._button_middle = button == "middle"
            viewer._last_mouse_x = 0.0
            viewer._last_mouse_y = 0.0
            viewer._mouse_move_callback(None, 24.0, 18.0)
        viewer._scroll_callback(None, 0.0, 1.0)

        self.assertTrue(np.isfinite(viewer._camera.azimuth))
        self.assertTrue(np.isfinite(viewer._camera.elevation))
        self.assertGreater(viewer._camera.distance, 0.0)

    def test_physical_defaults_match_current_crazyswarm_configuration(self):
        config = default_config()

        np.testing.assert_allclose(config.vehicle_masses, [0.0434, 0.0463, 0.0422])
        np.testing.assert_allclose(config.cable_lengths, [0.694, 0.692, 0.631])
        np.testing.assert_allclose(config.payload_size, [0.08, 0.06, 0.05])
        self.assertAlmostEqual(config.payload_mass, 0.054)
        self.assertAlmostEqual(config.prop_guard_outer_diameter, 0.052)
        self.assertAlmostEqual(config.ramp_position_hold_gain_scale, 2.0)
        np.testing.assert_allclose(
            config.attachment_points,
            [[0.04, 0.0, 0.025], [-0.04, 0.03, 0.025], [-0.04, -0.03, 0.025]],
        )

    def test_link_direction_feedback_uses_simulation_tuning_value(self):
        self.assertAlmostEqual(default_config().transport_link_gain_scale, 1.0)

    def test_z_down_conversion_is_an_involution(self):
        vector = np.array([0.3, -0.2, 1.7])
        np.testing.assert_allclose(z_down_to_up(z_down_to_up(vector)), vector)
        np.testing.assert_allclose(z_up_to_down(z_up_to_down(vector)), vector)
        np.testing.assert_allclose(z_down_to_up([1.0, 2.0, -3.0]), [1.0, 2.0, 3.0])

    def test_quintic_profile_has_zero_endpoint_velocity_and_acceleration(self):
        start = quintic_profile(0.0, 3.0)
        end = quintic_profile(3.0, 3.0)

        np.testing.assert_allclose(start, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(end, [1.0, 0.0, 0.0])

    def test_generated_scene_has_three_unilateral_cables(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            self.assertEqual(model.model.ntendon, 3)
            self.assertEqual(model.model.nq, 28)
            np.testing.assert_array_equal(model.model.tendon_limited, [1, 1, 1])
            np.testing.assert_allclose(model.model.tendon_range[:, 1], [0.694, 0.692, 0.631])
            np.testing.assert_allclose(model.model.site_pos[model._anchor_site_ids], 0.0)
            self.assertAlmostEqual(model.model.body_mass[model._load_body_id], 0.054)
        finally:
            model.close()

    def test_generated_scene_includes_non_colliding_propeller_guards(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            self.assertAlmostEqual(model.config.prop_guard_outer_diameter, 0.052)
            guard_ids = [
                mujoco.mj_name2id(
                    model.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{vehicle.name}_prop_guard_{rotor}_0",
                )
                for vehicle in model.config.vehicles
                for rotor in range(4)
            ]
            self.assertTrue(all(geom_id >= 0 for geom_id in guard_ids))
            np.testing.assert_array_equal(model.model.geom_contype[guard_ids], 0)
            np.testing.assert_array_equal(model.model.geom_conaffinity[guard_ids], 0)
            np.testing.assert_allclose(
                model.model.geom_size[guard_ids, 0],
                model.config.prop_guard_bar_radius,
            )
        finally:
            model.close()

    def test_tension_ramp_freezes_handoff_vehicle_and_payload_references(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            for qpos, target in zip(model._vehicle_qpos, model._takeup_targets, strict=True):
                model.data.qpos[qpos : qpos + 3] = target
            mujoco.mj_forward(model.model, model.data)
            model._set_phase(CrazySwarmSlungPhase.TENSION_RAMP)
            held_vehicles = model._ramp_vehicle_hold_positions.copy()
            held_payload = model._ramp_payload_hold_position.copy()

            for qpos in model._vehicle_qpos:
                model.data.qpos[qpos] += 0.05
            model.data.qpos[model._load_qpos + 2] += 0.01
            mujoco.mj_forward(model.model, model.data)
            positions, velocities = model._ramp_independent_reference()
            payload_target = model._payload_reference()

            np.testing.assert_allclose(positions, held_vehicles)
            np.testing.assert_allclose(velocities, 0.0)
            np.testing.assert_allclose(payload_target["position"], held_payload)
        finally:
            model.close()

    def test_controller_commands_respect_crazyswarm_limits(self):
        config = default_config()
        controller = CrazySwarmSlungController(config)
        model = CrazySwarmSlungModel(render=False, config=config)
        try:
            model.reset()
            commands = controller.independent_commands(
                model.vehicle_states(),
                model._takeoff_targets,
                np.zeros((3, 3)),
                config.timestep,
            )
            self.assertEqual(len(commands), 3)
            for command, vehicle in zip(commands, config.vehicles, strict=True):
                self.assertTrue(np.isfinite(command.thrust))
                self.assertGreaterEqual(command.thrust, 0.0)
                self.assertLessEqual(command.thrust, vehicle.max_command_thrust)
                self.assertTrue(np.all(np.isfinite(command.body_rate)))
                self.assertTrue(
                    np.all(np.abs(command.body_rate) <= np.asarray(vehicle.max_body_rate) + 1e-12)
                )
        finally:
            model.close()

    def test_csv_records_three_vehicle_rows_and_controller_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "slung.csv"
            model = CrazySwarmSlungModel(render=False, csv_path=csv_path)
            try:
                model.step()
                model.step()
            finally:
                model.close()

            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["vehicle_id"] for row in rows}, {"CF3", "CF4", "CF5"}
        )
        self.assertIn("cable_tension_n", rows[0])
        self.assertIn("payload_observed_velocity_x", rows[0])
        self.assertIn("payload_observation_derivatives_valid", rows[0])
        self.assertEqual(rows[0]["payload_state_estimator"], "observer")
        self.assertAlmostEqual(
            float(rows[0]["payload_velocity_filter_cutoff_hz"]), 3.0
        )
        self.assertIn("desired_tension_n", rows[0])
        self.assertIn("payload_kp_z", rows[0])
        self.assertAlmostEqual(float(rows[0]["prop_guard_outer_diameter_m"]), 0.052)
        self.assertAlmostEqual(float(rows[0]["prop_guard_bar_diameter_m"]), 0.002)
        self.assertAlmostEqual(float(rows[0]["ramp_position_hold_gain_scale"]), 2.0)
        self.assertEqual(rows[0]["flight_phase"], CrazySwarmSlungPhase.GROUND_SLACK.value)

    def test_csv_tendon_constraint_force_reports_cable_tension(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            for qpos in model._vehicle_qpos:
                model.data.qpos[qpos + 2] = 1.0
            mujoco.mj_forward(model.model, model.data)
            tensions = model._cable_tensions()
            self.assertTrue(np.all(tensions > 0.0))
        finally:
            model.close()

    def test_headless_run_completes_lift_hover_and_landing(self):
        model = CrazySwarmSlungModel(render=False)
        try:
            summary = model.simulate()
            self.assertEqual(summary["phase"], CrazySwarmSlungPhase.LANDED.value)
            self.assertTrue(summary["all_states_finite"])
            self.assertGreater(summary["max_payload_height"], 0.9)
            self.assertTrue(summary["payload_ground_contact"])
            self.assertTrue(summary["all_vehicles_ground_contact"])
            self.assertEqual(
                summary["phases_seen"],
                [phase.value for phase in CrazySwarmSlungPhase if phase is not CrazySwarmSlungPhase.EMERGENCY_LANDING],
            )
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
