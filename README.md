# Udaan CrazySwarm 三机吊运仿真

本项目在 Udaan 中实现了基于 MuJoCo 的三架 Crazyflie 协同吊运仿真。仿真从载荷和无人机位于地面开始，经过独立起飞、收绳、张力建立、协同抬升和悬停，最后完成载荷与无人机降落。控制器结构和物理参数参考 CrazySwarm MATLAB 吊运程序，并针对 MuJoCo 仿真进行调试。

## 安装

在项目根目录安装：

```bash
python -m pip install -e .
```

## 运行

```bash
# 打开 MuJoCo 窗口，执行完整流程并自动保存带时间戳的 CSV 到 logs/
udaan run crazyswarm-slung

# 无窗口运行，指定 CSV 文件
udaan run crazyswarm-slung --no-render --csv logs/hover-test.csv

# 限制仿真时长；省略 --time 会运行完整流程
udaan run crazyswarm-slung --time 61.2 --no-render --csv logs/short-run.csv

# 同时录制仿真画面和控制数据
udaan run crazyswarm-slung --record logs/run.mp4 --csv logs/run.csv

# 关闭 CSV 记录
udaan run crazyswarm-slung --no-csv
```

完整流程阶段为：`GROUND_SLACK`、`INDEPENDENT_TAKEOFF`、`TAKEUP`、`TENSION_RAMP`、`REFERENCE_LIFT`、`HOVER`、`LANDING_TAUT`、`LANDING_RELEASE`、`LANDED`。默认在 1 m 高度悬停 30 秒。

MuJoCo 窗口支持鼠标旋转、平移和滚轮缩放；按 `1` 恢复斜视，按 `2`、`3`、`4` 切换侧视或俯视，按 `F` 切换跟随载荷和自由观察，按 `Esc` 或 `Q` 关闭窗口。

## 绘制 CSV

绘图脚本可显示载荷位置误差与姿态、无人机轨迹、吊绳方向误差、张力、高度和推力。省略 CSV 路径时会读取 `logs/` 中最新的吊运记录。

```bash
# 绘制最新日志
python scripts/plot_crazyswarm_slung_csv.py

# 只绘制 HOVER 阶段并保存图片
python scripts/plot_crazyswarm_slung_csv.py logs/hover-test.csv \
  --phase HOVER --save logs/hover-test.png

# 保存图片但不打开绘图窗口
python scripts/plot_crazyswarm_slung_csv.py logs/hover-test.csv \
  --phase HOVER --save logs/hover-test.png --no-show
```

仿真 CSV 每个控制步为 CF3、CF4、CF5 各记录一行，包含飞行阶段、位置和姿态、目标与误差、控制指令、实际推力、吊绳状态和张力，以及本次运行使用的控制参数。

## 代码结构

- `udaan/crazyswarm_slung_parameters.py`：无人机、载荷、吊绳和控制器参数。
- `udaan/control/crazyswarm_slung.py`：独立起飞控制和协同吊运控制器。
- `udaan/models/mujoco/crazyswarm_slung.py`：MuJoCo 仿真、阶段状态机和数据采样。
- `udaan/models/mujoco/crazyswarm_slung_scene.py`：MJCF 场景生成。
- `udaan/models/mujoco/crazyswarm_slung_logging.py`：CSV 字段、记录器和姿态角转换。
- `scripts/plot_crazyswarm_slung_csv.py`：CSV 数据绘图。

## 当前仿真参数

- Crazyflie 机体外形：`50 x 50 x 14 mm`；臂长：`46.5 mm`。
- 螺旋桨保护罩视觉外径：`52 mm`。这是根据桨径估算的显示几何，不参与碰撞或质量计算。
- 载荷：质量 `54 g`，尺寸 `80 x 60 x 50 mm`。
- 三根吊绳长度：`0.694 m`、`0.692 m`、`0.631 m`。
- 仿真时间步长：`0.002 s`（500 Hz）；推力一阶时间常数：`0.012 s`。
- 载荷姿态带宽：roll/pitch/yaw 为 `6.0/6.0/0.30 Hz`。其中 yaw 当前仿真调试值为 `0.30 Hz`；MATLAB 参考配置为 `0.45 Hz`。
- `transport_link_gain_scale=0.01` 是当前 MuJoCo 仿真的绳向反馈调试比例，不是 Crazyflie 实机参数。

参数集中在 `udaan/crazyswarm_slung_parameters.py`，每次运行的实际参数也会写入 CSV，便于后续对照调参。

## 测试

```bash
python -m unittest tests.test_crazyswarm_slung -v
```

## 许可证

许可证信息见 [`LICENSE`](LICENSE)。
