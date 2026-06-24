# Stage7 外部基线启动与论文实验支撑分析

日期：2026-06-21  
项目：无人机视角火焰/烟雾小目标检测  
服务器目录：`/home/uav/gu/projects/FireAndSmoke_3`

## 1. 已读取与核对的资料

本次分析综合了以下资料：

- `D:\Researching\.workbuddy\memory\2026-06-21.md`
- `D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\reports\PAPER_EXPERIMENT_COMPLETION_PLAN_20260621.md`
- 服务器 Stage6 D2-D7 训练日志、`results.csv`、固定评估 CSV
- 近两年火灾/无人机小目标检测论文实验设置检索结果

检索到的近期相关论文显示，火灾/无人机小目标检测论文通常至少报告 mAP50、mAP50-95、Precision、Recall、Params、GFLOPs、FPS，并进行多模型基线对比和消融。例如 FirePM-YOLO 在 FireRescue 数据集上与 YOLOv12 等主流检测器对比并强调实时性；TCSN-YOLO 报告 APs、AP50、AP50:95、Params 与 GFLOPs；SPTD-YOLO 针对 UAV 小目标检测报告相对 YOLOv12 的 mAP 提升和效率表现。这说明我们当前计划中的外部基线、效率指标和小目标指标是论文投稿前必须补齐的核心实验。

参考文献检索来源：

- FirePM-YOLO: Position-Enhanced Mamba for YOLO-Based Fire Rescue Object Detection from UAV Perspectives, Sensors, 2026, DOI: `10.3390/s26072064`
- TCSN-YOLO: A Small-Target Object Detection Method for Fire Smoke, Fire, 2025, DOI: `10.3390/fire8120466`
- SPTD-YOLO: Small-Object-Aware Pyramidal and Task-Aligned Dynamic YOLO for UAV Small Object Detection, Applied Sciences, 2026, DOI: `10.3390/app16126062`

## 2. 当前主模型状态

D5 与 D7 已形成稳定复现实验。D7 使用与 D5 相同配置，仅修改 `seed=1`，最终主验证集与固定评估均与 D5 一致。

| 模型 | seed | val mAP50 | val mAP50-95 | public hard recall@0.20 | tiny recall@0.20 | FP/image@0.20 | false_alarm_v2 FP/1000@0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D5 | 0 | 0.83146 | 0.57379 | 0.87038 | 0.71973 | 0.520 | 5 |
| D7 | 1 | 0.83146 | 0.57379 | 0.87038 | 0.71973 | 0.520 | 5 |

这说明当前 D5 策略具有很强的可重复性。由于 D5/D7 已完全一致，现阶段继续补 seed=2 的收益低于补外部基线。论文主模型建议暂定为 D5/D7 对应的 S5+hard-negative x4 冻结校准策略。

## 3. D6 结论与论文定位

D6 从 D5 best 出发，将输入尺寸从 960 提高到 1280，但结果明显退化：

| 模型 | imgsz | val mAP50-95 | public hard tiny recall@0.20 | FP/image@0.20 | false_alarm_v2 FP/1000@0.20 |
|---|---:|---:|---:|---:|---:|
| D5 | 960 | 0.57379 | 0.71973 | 0.520 | 5 |
| D6 | 1280 | 0.48266 | 0.71525 | 0.530 | 6 |

D6 触发早停，最佳结果出现在第 1 个 epoch。该结果表明，单纯提高输入分辨率并不能改善当前 UAV 火焰/烟雾检测任务，反而破坏主验证集定位质量。论文中可以将 D6 作为分辨率负向消融，支撑“改进主要来自 P2、小目标切片数据与 hard negative 校准，而非简单依赖高分辨率”的论点。

## 4. 当前实验对论文的支撑点

现有实验已经可以支撑以下论文主张：

1. P2 小目标检测头与切片数据策略对 UAV 小目标火焰/烟雾检测有必要性。
2. Hard negative mining 可以把误报从 V4 的 9 FP/1000 降至 D5/D7 的 5 FP/1000。
3. 多目标评估协议比单一 mAP 更适合早期火灾预警任务，因为 D5 不是最高 mAP 的单点模型，却在低误报和困难场景 FP/image 上更适合部署。
4. 阈值策略可形成 recall-first、balanced、conservative 三个操作点，为实际部署提供可解释选择。
5. D6 负向实验说明更高输入分辨率不是稳定改进方向。

目前仍缺少的硬证据是：外部基线、Params/GFLOPs/FPS、per-size AP、视频级评估。这些缺口如果不补，论文很难达到 SCI 一区/二区或顶会实验标准。

## 5. 已启动的下一步：Stage7 YOLOv8 外部基线

根据 `PAPER_EXPERIMENT_COMPLETION_PLAN_20260621.md` 的 P0 优先级，服务器已启动 Stage7 外部基线训练。第一批先训练 YOLOv8n、YOLOv8s、YOLOv8m 三个官方模型，统一使用当前主数据集：

```text
数据集: /home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml
输出目录: /home/uav/gu/projects/FireAndSmoke_3/runs_stage7_baselines
日志: /home/uav/gu/stage7_yolov8_baselines_train.log
脚本: /home/uav/gu/projects/FireAndSmoke_3/tools/stage7_train_yolov8_baselines_server4090.sh
```

启动状态：

- 后台 PID：`422950`
- 当前模型：`yolov8n.pt`
- 当前 run：`stage7_yolov8n_stage4_i960_e100`
- 双卡 DDP 已启动
- AMP 检查通过
- 已进入第 1/100 个 epoch
- 当前显存约 9.5GB/卡
- 暂无 OOM、NaN、Traceback 或训练中断

训练配置：

| 模型 | imgsz | epochs | patience | batch | optimizer | lr0 |
|---|---:|---:|---:|---:|---|---:|
| YOLOv8n | 960 | 100 | 20 | 64 | AdamW | 0.001 |
| YOLOv8s | 960 | 100 | 20 | 48 | AdamW | 0.001 |
| YOLOv8m | 960 | 100 | 20 | 32 | AdamW | 0.001 |

查看实时日志：

```bash
tail -f /home/uav/gu/stage7_yolov8_baselines_train.log
```

## 6. Stage7 完成后的固定评估要求

每个外部基线完成后都必须执行同一套评估：

1. 主验证集：读取 `results.csv` 中 best mAP50、mAP50-95、Precision、Recall。
2. public hard：计算 recall、tiny recall、FP/image。
3. false_alarm_v2：计算 FP/1000。
4. 效率指标：Params、GFLOPs、FPS、模型大小。

固定评估示例：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
bash tools/stage6_eval_model_server.sh runs_stage7_baselines/stage7_yolov8n_stage4_i960_e100/weights/best.pt
```

## 7. 后续执行顺序

当前优先级如下：

1. 等 Stage7 YOLOv8n/s/m 完成，并逐一跑固定评估。
2. 若 YOLOv8m 基线完成且服务器稳定，再补 YOLOv10m 或 YOLOv11/YOLOv12 中可用版本。
3. 采集 D5/D7、V4/S5/No-P2、YOLOv8n/s/m 的 Params、GFLOPs、FPS。
4. 实现 per-size AP 或至少输出 tiny/small/overall recall 的统一分解。
5. 启动视频级评估，报告 FP/min、首次检测时间和时序稳定性。

主线判断：D5/D7 已足够作为当前主模型候选；现在最需要补的是论文对比实验，而不是继续微调主模型。
