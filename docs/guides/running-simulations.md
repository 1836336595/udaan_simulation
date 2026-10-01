# Running simulations

This page summarises the CLI and scripted entry points. Every command accepts
`--help` for the full option list.

## Model capabilities

Each model ships with several backends and input modes. The table below shows
what each supports — use it to pick the right entry point for your task.

| Model | `base` (dynamics only) | `vfx` (VPython) | `mujoco` (MuJoCo) | Fleet |
|---|:---:|:---:|:---:|:---:|
| {py:class}`~udaan.models.quadrotor.QuadrotorBase` | ✓ | ✓ | ✓ | ✓ (via `fleet`) |
| {py:class}`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase` | ✓ | ✓ | ✓ (tendon / links / cable) | ✓ (via `cspayload-fleet`) |
| {py:class}`~udaan.models.mujoco.MultiQuadrotorCSPointmass` | — | — | ✓ | — |
| {py:class}`~udaan.models.mujoco.MultiQuadRigidbody` | — | — | ✓ | — |

See {doc}`controllers` for the controllers shipped with each model and how to swap them.

### Input and force types

`QuadrotorBase` and `QuadrotorCsPayloadBase` accept several input
repackagings, chosen at construction time via the `input=...` kwarg:

| Input type | Controller produces | Integrator consumes | Use when |
|---|---|---|---|
| `acceleration` (default) | 3-vec desired thrust force | wrench via geometric attitude controller | high-level trajectory tracking |
| `wrench` | 4-vec `[f, M_x, M_y, M_z]` | wrench directly | custom attitude laws, SysID |
| `prop_forces` | 4-vec per-rotor forces | allocated wrench | motor-level experiments |

### Cable models (payload only)

| `cable_model` | MuJoCo backend | Captures slack? | Notes |
|---|---|:---:|---|
| `tendon` | spatial tendon constraint | — | fast, less realistic under slack |
| `links` (default) | rigid N-link chain | ✓ | most stable, recommended default |
| `cable` | MuJoCo composite cable | ✓ | experimental; see {doc}`../theory/dynamics/quadrotor-cspayload` caveats |

## Quadrotor

```bash
udaan run quadrotor                              # MuJoCo viewer, hover
udaan run quadrotor -m base                      # pure dynamics, no rendering
udaan run quadrotor --traj spiral -p 0,0,2       # helical spiral trajectory
udaan run quadrotor --traj lissajous -p 0,0,2    # 3D Lissajous
```

## Quadrotor with cable-suspended payload

```bash
udaan run quad-payload -c tendon        # spatial-tendon cable model
udaan run quad-payload -c links         # N-link rigid cable model
```

The controller used is derived in {doc}`../theory/controllers/quadrotor-payload`; the
underlying dynamics are covered in {doc}`../theory/dynamics/quadrotor-cspayload`.

## Side-by-side fleets

Two fleet commands exist for comparing controllers or gains:

```bash
udaan run fleet --demo l1-comparison           # N quadrotors, L1 vs PD
udaan run cspayload-fleet --demo gain-sweep    # N quad+payload, gain sweep
```

## CrazySwarm 三机吊运

该场景从载荷和三架无人机位于地面开始，依次执行独立起飞、收绳、协同抬升、30 秒悬停、载荷着陆和无人机着陆。

```bash
udaan run crazyswarm-slung                    # 完整仿真并打开 MuJoCo 窗口
udaan run crazyswarm-slung --no-render        # 无窗口运行完整流程
udaan run crazyswarm-slung --time 20 --no-render  # 最多运行 20 秒
udaan run crazyswarm-slung -r slung.gif       # 录制完整仿真
udaan run crazyswarm-slung --csv logs/tune.csv --no-render  # 指定 CSV 路径
udaan run crazyswarm-slung --no-csv --no-render  # 关闭 CSV
```

默认自动在 `logs/` 下生成带时间戳的 CSV，每个仿真步为 CF3、CF4、CF5 各写一行。字段包括阶段、机体和载荷位置/速度/姿态、参考值与误差、推力和角速度指令、绳长/绳速/实际与期望张力，以及本次运行使用的控制器增益；仿真中断时最多丢失最近约 25 行以内尚未刷新的记录。

使用 `scripts/plot_crazyswarm_slung_csv.py` 绘制三维轨迹、载荷位置误差与姿态、吊绳方向误差、张力、高度和推力。省略 CSV 路径时自动选择 `logs/` 中最新的吊运日志；`--phase` 可只看指定阶段，`--save` 可保存 PNG/PDF：

```bash
python scripts/plot_crazyswarm_slung_csv.py
python scripts/plot_crazyswarm_slung_csv.py logs/tune.csv --phase HOVER --save hover.png
```

窗口中可左键拖动旋转、右键拖动平移、滚轮缩放；按 `1` 恢复斜视，按 `2`/`3` 切换侧向视角，按 `4` 俯视。按 `F` 可在跟随载荷和自由观察之间切换，`Esc` 或 `Q` 关闭窗口。

默认参数取自 CrazySwarm 当前工作树中的机体和吊运配置，并使用 MATLAB 刚体载荷控制方程。可通过 Python API 获取运行摘要或替换场景参数：

依据最新 CSV 的同条件悬停对照，仿真默认 `transport_link_gain_scale=0.01`：载荷横向位置 RMS 从 `0.080 m` 降至 `0.073 m`，三根吊绳方向误差的 95 分位也略有下降。试验值 `0.05` 的误差更大。CrazySwarm 原始值为 `0.0`；该比例仅为仿真调参，其余控制参数保持不变。

`TENSION_RAMP` 参考冻结在绳索交接瞬间的载荷和无人机位置；5 秒张力渐增使用五次平滑曲线，并以 `ramp_position_hold_gain_scale=2.0` 保留独立位置环反馈。完整对照运行中，CF3/CF4/CF5 ramp 最大偏移约为 `2.3/1.4/1.1 cm`，最小机间距约 `23.3 cm`。当前最新 CSV 中 yaw 自然频率为 `0.45 Hz`，与 MATLAB 的 `wnLoad = 2*pi*[6, 6, 0.45]` 一致。桨叶保护罩按视觉代理绘制：外径 `52 mm`、杆径 `2 mm`；Bitcraze 官方页面没有给出可核实的外形尺寸，因此该尺寸按现有 `46 mm` 桨径推算，不参与碰撞或质量计算。

```python
from udaan.models.mujoco import CrazySwarmSlungModel

model = CrazySwarmSlungModel(render=False)
summary = model.simulate()
print(summary["phase"], summary["max_payload_height"])
model.close()
```

Python API 可通过 `csv_path="logs/tune.csv"` 指定日志文件；不指定时不创建 CSV。

## Scripted (Python)

```python
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

mdl = QuadrotorCsPayloadMujoco(render=True, cable_model="links")
mdl._payload_controller.setpoint = lambda t: ([0, 0, 1], [0, 0, 0], [0, 0, 0])
mdl.simulate(tf=8.0, payload_position=[1, 1, 0.5])
```
