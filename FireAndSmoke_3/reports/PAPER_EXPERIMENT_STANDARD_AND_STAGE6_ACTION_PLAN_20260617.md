# 论文实验标准与 Stage6 执行方案

日期：2026-06-17

项目：面向无人机视角小目标火焰与烟雾检测的 YOLO 模型优化。

本文档把当前工程结果整理成论文实验路线。目标不是单纯提高 mAP，而是建立一套能支撑 SCI 一区/二区论文写作的实验链条：任务难点明确、数据来源可追溯、基线充分、消融清楚、误报分析完整，并且能说明部署效率。

## 1. 顶刊顶会与高质量期刊论文的实验要求

近期火焰/烟雾检测、无人机小目标检测论文通常会包含以下实验模块。

1. 数据集与划分协议
   - 需要说明数据规模、类别分布、训练/验证/测试划分、图像来源、标注质量控制，以及是否存在数据泄漏。
   - FASDD 是重要参考，因为它强调多源火焰/烟雾图像、无人机/遥感场景、光照变化、视角变化和质量控制。
   - D-Fire 对我们也有参考意义。它包含大量空场景/负样本，说明火灾检测任务不能只看正样本，必须专门评估背景误报。

2. 强基线对比
   - 不能只比较 V4、S5、No-P2 这几个内部模型。论文需要加入多个 YOLO 版本或不同规模模型，条件允许时还应加入 RT-DETR、Faster R-CNN 等非 YOLO 基线。
   - 无人机火焰/烟雾检测论文常见指标包括 Precision、Recall、mAP50、mAP50-95、FPS、Params、FLOPs、延迟，以及小目标 AP 或小目标召回。

3. 消融实验
   - 每个方法贡献都要单独打开/关闭：P2 检测头、hard-negative 挖掘、切片推理/切片微调、训练策略、阈值与后处理。
   - 只报告最终模型指标不够，必须说明每个模块带来的收益和代价。

4. 鲁棒性与部署分析
   - 需要报告复杂背景下的误报和漏检，例如云雾、烟雾状背景、强反光、地形纹理、小而远的火点、视频帧抖动。
   - 如果论文强调无人机或边缘部署，必须报告目标硬件或近似硬件上的 FPS、延迟和模型大小。

5. 失败案例分析
   - 高质量论文不会回避误报。需要展示 FP/FN 案例，并解释错误来源：云/雾、太阳眩光、地表纹理、火焰像素过少、烟雾透明、运动模糊或标注歧义。

## 2. 当前已有的论文支撑

### 2.1 主模型与 P2 消融证据

已同步结果显示，V4 YOLOv8m-P2 仍是当前最稳定的主参考模型。

| 模型 | Val mAP50 | Val mAP50-95 | Holdout mAP50 | Holdout mAP50-95 |
|---|---:|---:|---:|---:|
| V4 P2 | 0.84371 | 0.58371 | 0.92376 | 0.68120 |
| S5 微调 | 0.83829 | 0.57902 | 0.92841 | 0.68532 |
| No-P2 消融 | 0.84082 | 0.58018 | 0.91585 | 0.67743 |

当前可以写进论文的关键结论：

- P2 检测层提高了 tiny fire 召回。已有 tiny fire Recall50 对比为：V4 P2 = 0.91014，No-P2 = 0.83641，提升 0.07373。

这个结论有清晰的消融支撑，适合作为论文方法贡献之一。

### 2.2 public hard holdout 阈值扫描

已完成文件：

- `reports/stage6_threshold_scan_v4_public_hard.csv`
- `reports/stage6_threshold_scan_s5_public_hard.csv`
- `reports/stage6_threshold_scan_nop2_public_hard.csv`

在 500 张 public hard holdout 图像上，使用 `conf=0.20`、`imgsz=960`、`IoU=0.50`：

| 模型 | Recall | Tiny Recall | Fire Recall | Smoke Recall | FP/image | Precision-like |
|---|---:|---:|---:|---:|---:|---:|
| V4 P2 | 0.86486 | 0.70852 | 0.83959 | 0.92523 | 0.542 | 0.85264 |
| S5 微调 | 0.87148 | 0.72197 | 0.84742 | 0.92897 | 0.538 | 0.85452 |
| No-P2 | 0.86486 | 0.70628 | 0.84038 | 0.92336 | 0.554 | 0.84986 |

结论：

- S5 在 public hard 阈值协议下略优于 V4，但提升幅度很小，不能写成决定性突破。
- No-P2 在 `conf=0.20` 时接近 V4，但阈值升高后 tiny recall 更弱。这说明 P2 是必要的，但仅靠 P2 还不能完全解决误报和漏检。
- 论文中应把 S5 写成微调候选模型，而不是主贡献模型。

### 2.3 固定误报评估集

已完成文件：

- `reports/stage6_threshold_scan_v4_false_alarm.csv`
- `reports/stage6_threshold_scan_s5_false_alarm.csv`
- `reports/stage6_threshold_scan_nop2_false_alarm.csv`
- `reports/stage6_threshold_scan_v4_false_alarm_v2.csv`
- `reports/stage6_threshold_scan_s5_false_alarm_v2.csv`
- `reports/stage6_threshold_scan_nop2_false_alarm_v2.csv`

在 `datasets/stage6_eval/uav_false_alarm_v1` 上，`conf=0.20`：

| 模型 | 图像数 | FP/100 images | FP/image | 误报类别 |
|---|---:|---:|---:|---|
| V4 P2 | 100 | 2 | 0.02 | smoke=2 |
| S5 微调 | 100 | 1 | 0.01 | smoke=1 |
| No-P2 | 100 | 3 | 0.03 | fire=1, smoke=2 |

v1 只有 100 张图，适合作为先导固定误报集，但不足以作为最终论文主证据。

`uav_false_alarm_v2` 已从 `fasdd_uav_curated/test` 的空标签图像中构建完成：

- 空标签候选图像：1997 张
- 固定评估集：1000 张
- 输出位置：`datasets/stage6_eval/uav_false_alarm_v2`
- 完整性检查：1000 images、1000 labels、1000 empty labels、0 non-empty labels

在 `uav_false_alarm_v2` 上，使用 `conf=0.20`、`imgsz=960`、`IoU=0.50`：

| 模型 | 图像数 | FP/1000 images | FP/image | 误报类别 |
|---|---:|---:|---:|---|
| V4 P2 | 1000 | 9 | 0.009 | fire=1, smoke=8 |
| S5 微调 | 1000 | 5 | 0.005 | smoke=5 |
| No-P2 | 1000 | 15 | 0.015 | fire=4, other=3, smoke=8 |

更新后的判断：

- S5 目前是误报控制最好的 balanced 候选。
- V4 仍是主参考模型，因为它的 P2 小目标证据稳定。
- No-P2 应保留为消融模型，不适合作为最终主模型候选；它在更大的误报集上更弱，同时削弱 tiny-object 论证。

### 2.4 Stage6 scout 结果

aggressive scout 路线失败，不应作为最终方法。

conservative scout 提升了召回，但 public hard 上误报显著增加：

| 模型 | Tiny Recall | Fire Recall | Smoke Recall | Public-hard FP total |
|---|---:|---:|---:|---:|
| V4 P2 | 0.70852 | 0.83959 | 0.92523 | 271 |
| Conservative scout | 0.77354 | 0.86150 | 0.93640 | 590 |

结论：

- 模型可以通过训练变得更敏感，尤其对 tiny fire/smoke 更敏感，但 FP 会明显上升。
- 下一轮模型选择必须采用多目标门槛，不能只看 mAP 或召回。

## 3. 与 SCI 一区/二区论文标准相比的不足

1. 误报评估集已经扩大，但还缺少背景分层。
   - `uav_false_alarm_v1` 有 100 张空标签图。
   - `uav_false_alarm_v2` 已有 1000 张 FASDD UAV curated test 空标签图。
   - 还需要按背景类型继续分层，例如云雾、强反光、道路/建筑、土壤/岩石、水面反射、工业烟雾状纹理、夜间或低光照。

2. 外部基线不足。
   - 当前内部对比包括 V4、S5、No-P2，已经能支撑消融和阶段性结论。
   - 论文目标还需要加入 YOLOv8n/s/m/l、YOLOv10 或 YOLOv11 系列、RT-DETR 或 Faster R-CNN，以及轻量化边缘模型。

3. 消融还不完整。
   - 已支持：P2 vs No-P2。
   - 仍需补充：P2 + hard negatives、P2 + sliced inference/fine-tuning、P2 + 阈值策略、P2 + conservative fine-tuning。

4. 视频指标还不足。
   - 当前前端能做视频检测，但论文需要量化：FP/min、漏检事件数、首次检测时间、时序稳定性和检测持续性。

5. 还需要效率与部署表。
   - 应报告 Params、FLOPs、FPS、延迟和模型大小。
   - 至少需要本机 GPU 和目标服务器/边缘设备之一的推理效率结果。

6. 统计可靠性还不够。
   - 最终主模型和关键基线最好进行 3 个随机种子训练，或对固定测试集做 bootstrap 置信区间。

## 4. Stage6 优化路线

### Step A：固定误报集扩展

当前状态：

- 已完成初步扩展：`datasets/stage6_eval/uav_false_alarm_v2`，共 1000 张 verified empty-label 图像。

下一步：

- 做背景类型分层。
- 条件允许时，加入人工确认或可信自动筛选的空视频帧。

约束：

- 只使用空 YOLO 标签图或人工确认空场景。
- 未经确认的视频帧不能直接当作负样本。
- 固定误报评估集不得进入训练。

验收条件：

- `missing_labels=0`
- `instances=0`
- 有 source manifest
- 所有图像都有空 label 文件

### Step B：正式 Stage6 D1 训练

正式命令已记录在 `configs/stage6_experiment_matrix.yaml`，本机 3070 版本已降 batch 启动。

核心设置：

- 数据：`datasets/stage6_mixed_3cls_union/data.yaml`
- 初始化：V4 best
- 架构：YOLOv8m-P2 3-class
- 训练策略：低学习率、冻结前层、弱增强、关闭 aggressive erasing

模型选择门槛：

| 门槛 | 最低要求 |
|---|---|
| public hard tiny recall | 不低于 V4 的 0.70852，目标不低于 0.75 |
| public hard FP/image | 不高于 V4 的 0.542，若超过则只能标为 recall-first 模型 |
| false-alarm v1 FP/100 | 不高于 V4 的 2 |
| false-alarm v2 FP/1000 | balanced 模型不高于 S5 的 5；可接受模型不高于 V4 的 9 |
| holdout mAP50-95 | 相比 V4 下降不超过 0.01 |
| smoke recall | 不低于 V4 public hard 的 0.92523 |

### Step C：阈值与后处理研究

对 V4、S5、D1 统一扫描：

- `conf = 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50`
- 指标包括 recall、tiny recall、fire recall、smoke recall、FP/image、precision-like
- 输出三个部署工作点：
  - recall-first：`conf=0.15-0.20`
  - balanced：当前前端默认 `conf=0.25`
  - conservative alarm：`conf=0.30-0.40`

### Step D：切片推理 / SAHI 实验

SAHI 直接面向小目标检测，对无人机远距离火点和早期烟雾有参考价值。

实验内容：

- 比较 full image inference 与 sliced inference。
- 在 public hard holdout 上统一阈值和 NMS。
- 报告 tiny recall、FP/image、AP/mAP。
- 同时报告延迟或 FPS，因为切片通常会提升召回，但会降低速度。

### Step E：视频压力测试

对 V4、S5、D1 分别测试：

- 视频来源：`D:\Researching\Yolo\Yolo\docs`
- 指标：
  - FP/min
  - 首次检测时间
  - 可见火焰/烟雾事件漏检数
  - 时序闪烁次数
  - 代表性成功与失败帧

## 5. 论文贡献表述建议

可以考虑以下贡献线：

1. 面向无人机早期火灾预警的小目标火焰/烟雾检测流程。
2. 引入 P2 检测层的 YOLOv8m，并用 No-P2 消融验证其对 tiny fire 召回的贡献。
3. 构建 verified-empty hard-negative mining 与固定 false-alarm 评估协议。
4. 用多目标门槛选择模型，同时考虑 tiny recall、smoke recall、FP/image、mAP 与部署效率。

暂时不要写成结论的内容：

- 不要声称 Stage6 scout 降低了误报。
- 不要把 S5 写成决定性优于 V4，除非正式 D1 和完整基线验证后仍成立。
- 不要只用 100 张 false-alarm v1 作为最终误报证据。
- 在外部基线、视频指标和效率指标补齐前，不要声称实验已经达到 SCI 一区/二区完整标准。

## 6. 本方案参考的论文与资源

- Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection", ICIP 2022 / arXiv:2202.06934. https://arxiv.org/abs/2202.06934
- D-Fire 数据集仓库，包含火焰/烟雾/空场景数据。https://github.com/gaia-solutions-on-demand/DFireDataset
- "An Improved Wildfire Smoke Detection Based on YOLOv8 and UAV Images", Sensors 2023. https://www.mdpi.com/1424-8220/23/20/8374
- "Multiscale wildfire and smoke detection in complex drone forest environments based on YOLOv8", Scientific Reports 2025. https://www.nature.com/articles/s41598-025-86239-w
- "YOLOFM: an improved fire and smoke object detection algorithm based on YOLOv5n", Scientific Reports 2024. https://www.nature.com/articles/s41598-024-55232-0
- FASDD: "An open flame and smoke detection dataset for deep learning in fire detection", DOI: 10.1080/10095020.2024.2347922。
