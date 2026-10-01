# CrazySwarm Three-UAV Sling-Load Simulation Design

## Goal

Add a Udaan/MuJoCo scenario that starts with three Crazyflies and a rectangular payload on the ground, independently lifts the aircraft, takes up three cables, transfers to cooperative payload control, holds a hover, then lowers the load and lands the aircraft.

## Current Context

Udaan's existing `MultiQuadrotorCSPointmass` model uses a shared point-mass payload and generic quadrotor defaults. It does not represent the CrazySwarm payload geometry, per-aircraft cable lengths, or the ground/slack/handoff/landing sequence.

CrazySwarm already contains a MATLAB rigid-payload simulator with independent takeoff, cable take-up, tension ramp, cooperative transport, and release/landing stages. Its default physical values differ from the current hardware configuration, so this design uses the live CrazySwarm YAML for hardware-specific values and the MATLAB project for missing geometry and control equations.

## Scope

- Add a dedicated Udaan CLI command: `udaan run crazyswarm-slung`.
- Simulate exactly three Crazyflie vehicles and one rigid rectangular payload in a MuJoCo scene with a floor and three individually sized, tension-only cables.
- Preserve the existing Udaan commands and models.
- Keep the CrazySwarm repository read-only; Udaan owns the port and its standalone defaults.
- Exclude lateral transport trajectories, mocap/ROS integration, radio/firmware emulation, and payload obstacles.

## Parameters

The command's default configuration mirrors the current CrazySwarm branch:

| Parameter | Value/source |
| --- | --- |
| Vehicles | 3, ordered CF3, CF4, CF5 |
| Vehicle mass | 0.0434, 0.0463, 0.0422 kg from `config/ctbr_vehicle.yaml` |
| Vehicle command thrust cap | 1.00 N per vehicle; calibrated total capability 1.176798 N |
| Body-rate limit | [3, 3, 2] rad/s; maximum tilt 10 degrees |
| Vehicle display geometry | 0.050 x 0.050 x 0.014 m body, 0.0465 m arm, 0.023 m rotor radius from MATLAB parameters |
| Payload | 0.054 kg, 0.080 x 0.060 x 0.050 m from `config/slung_payload.yaml` |
| Cable lengths | [0.694, 0.692, 0.671] m in CF3/CF4/CF5 order |
| Attachments | +x edge midpoint, -x/+y corner, -x/-y corner; all on the top face |
| Payload inertia | Homogeneous-box inertia derived from the selected mass and dimensions |
| Controller rate | 500 Hz, matching the MATLAB `dt = 0.002 s` baseline |

Per-aircraft position, velocity, integral, independent-takeoff, transport-attitude, thrust, tilt, and rate settings are copied from the current `ctbr_vehicle.yaml` entries. Payload position gains, attitude bandwidth/damping, tension allocation, and cable-direction gains are copied from `slung_payload.yaml` and its MATLAB source. The current `transport_link_gain_scale: 0.0` is honored, so cable-direction feedback starts disabled as it does in the live configuration; changing that scale remains a config choice, not a new controller. Values absent from the live YAML, including vehicle inertias and visual geometry, use the CrazySwarm MATLAB parameter file. The MATLAB-only 0.0325 kg vehicle mass, 0.080 kg payload mass, and 0.65 m uniform cable length are not used as defaults.

The simulation uses Udaan/MuJoCo's world-z-up convention. MATLAB z-down quantities are converted once at the configuration boundary; controller internals and rendered geometry use z-up thereafter.

## Control And Phase Flow

The model exposes collective thrust and body-rate commands with CrazySwarm's per-aircraft saturation. A bounded rate-response model follows the MATLAB Crazyflie inner-loop approximation; MuJoCo integrates rigid-body motion, contacts, and unilateral cable constraints. The tension-ramp phase blends independent and MATLAB-derived transport commands; cable tension itself comes from MuJoCo's unilateral cable constraints.

1. `GROUND_SLACK`: the rectangular payload rests on the floor and the cables are slack.
2. `INDEPENDENT_TAKEOFF`: each vehicle follows the CrazySwarm independent CTBR position controller to its 0.50 m staging height over 5.0 s.
3. `TAKEUP`: the aircraft move to their cable-length geometry over 8.0 s; handoff requires cable-distance confirmation for 0.50 s and a further 0.30 s valid-state hold.
4. `TENSION_RAMP`: blend from independent control into the MATLAB cooperative transport controller over 5.0 s.
5. `REFERENCE_LIFT`: raise the load reference over 12.0 s to the configured 1.0 m hover height.
6. `HOVER`: hold the load at the target for 30.0 s.
7. `LANDING_TAUT`: lower the load while maintaining cooperative cable control for the configured 55% approach fraction; use the configured 4.0 s minimum descent duration and 0.25 m/s speed cap.
8. `LANDING_RELEASE`: after load contact, release the cable constraint and independently lower the aircraft; finish in `LANDED` after ground contact and 0.30 s settle confirmation.

The timings and thresholds above come from current CrazySwarm `slung_payload.yaml` and `ctbr_controller.yaml`. Handoff also requires cable distance within 0.01 m and cable angle within 10 degrees. A payload state may be held for 0.20 s; a longer fault triggers CrazySwarm's bounded emergency landing after 0.30 s. Invalid or non-finite states never permit transport handoff.

## Components

- `udaan/models/mujoco/crazyswarm_slung.py` owns the three-vehicle rigid-payload MJCF, state extraction, simulation stepping, phase transitions, and a final run summary. MJCF is generated in memory from the configuration so per-vehicle cable lengths do not require a checked-in generated file.
- `udaan/control/crazyswarm_slung.py` owns independent CTBR position control and the MATLAB cooperative payload, tension-allocation, cable-direction, and attitude control port.
- `udaan/cli/run.py` exposes the command and existing recording/rendering options.
- `tests/test_crazyswarm_slung.py` covers configuration, stage references/transitions, and a headless MuJoCo end-to-end run.
- `README.md` and `docs/guides/running-simulations.md` document the command and parameter provenance.

## Verification

- Unit tests verify the ordered hardware parameters, box inertia, z-axis conversion, and phase-reference boundary conditions.
- State-machine tests verify that phases progress in order, handoff does not occur before cable geometry and confirmation conditions are satisfied, and landing finishes only after floor contact.
- A headless MuJoCo integration run verifies finite states, bounded commands, cable lengths within their unilateral constraints during taut phases, payload lift to the hover target, and final ground contact for the payload and all vehicles.
- Run the focused test module, the full Udaan test suite, and a rendered CLI smoke test with the default scenario.
