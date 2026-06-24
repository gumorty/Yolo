# FireAndSmoke_3

第三轮火焰/烟雾检测训练工程，目标是从“页面展示模型”转向“解决无人机真实巡检中的小目标、多目标、弱烟和红外热成像预警问题”。

## 核心变化

- 使用 YOLOv8m-P2 结构，增加 P2/4 检测头，提高远处小火苗、小烟雾的特征保留能力。
- 不再默认混合 Sensors 与自有 fire_pic 数据，先建立严格数据审计和 Sensors-only 小目标基线。
- 保留 2类和 3类两条训练线，3类继续使用 fire / other / smoke 来研究误报抑制。
- 增加 tiny/small/medium/large 小目标召回评估脚本。
- 增加 RGB-Thermal 数据配对审计脚本，为真实红外热成像融合训练做准备。

## 目录

- `models/`：YOLOv8m-P2 2类/3类模型 YAML。
- `tools/`：数据审计、小目标评估、RGB-T 配对审计、视频抽帧工具。
- `scripts/`：目标主机 PowerShell 执行脚本。
- `reports/`：本轮设计、文献映射、训练命令和本地复盘。

## 推荐先读

1. `reports/LOCAL_ARTIFACT_REVIEW.md`
2. `reports/LITERATURE_METHOD_MAP.md`
3. `reports/THIRD_ROUND_DESIGN.md`
4. `reports/TRAINING_COMMANDS.md`

## 当前主线

先训练：

- `stage3_yolov8m_p2_sensors3_e120`
- `stage3_yolov8m_p2_sensors2_e120`

不要急着训练 300 epoch。先用 960 输入尺寸完成 P2 基线，再用小目标召回和外部视频测试决定是否上 1280、切片训练或 RGB-T 双流模型。
