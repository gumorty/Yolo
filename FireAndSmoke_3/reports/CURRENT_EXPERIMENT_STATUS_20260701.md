# 当前实验状态汇总

日期：2026-07-01  
项目：无人机视角火焰与烟雾小目标检测  
核查来源：本地 Stage6/Stage7 报告、训练脚本、配置文件、模型 YAML，以及服务器 `/home/uav/gu/projects/FireAndSmoke_3` 下的训练日志和结果文件。

## 1. 当前总体状态

当前实验已经不处于 Stage6/Stage7 主训练进行中状态。服务器端已经完成 Stage6 D2-D7 内部迭代链，以及 Stage7 YOLOv8n/s/m 外部基线训练。2026-07-01 核查时，服务器上没有发现属于本项目的 `stage6`、`stage7`、`train_yolo` 或 Ultralytics 训练进程。两张 RTX 4090 当时被其他 Python 进程占用，因此后续新增训练或大规模评估需要等待 GPU 释放。

服务器上最新关键实验产物包括：

- Stage6 D7 可重复性实验：`/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d7_s5_hardneg_balanced_x4_i960_seed1`
- Stage7 YOLOv8 外部基线：
  - `/home/uav/gu/projects/FireAndSmoke_3/runs_stage7_baselines/stage7_yolov8n_stage4_i960_e100`
  - `/home/uav/gu/projects/FireAndSmoke_3/runs_stage7_baselines/stage7_yolov8s_stage4_i960_e100`
  - `/home/uav/gu/projects/FireAndSmoke_3/runs_stage7_baselines/stage7_yolov8m_stage4_i960_e100`
- 固定评估日志：
  - `/home/uav/gu/stage7_eval_v8n.log`
  - `/home/uav/gu/stage7_eval_v8s.log`
  - `/home/uav/gu/stage7_eval_v8m.log`
  - `/home/uav/gu/stage6_d7_eval.log`

## 2. 当前训练流程

当前实验流程不是单次 YOLO 训练，而是围绕无人机小目标火烟检测逐步构建的多阶段流程。

Stage4/S5 阶段以修正后的三类别 fire/other/smoke 数据集为基础，形成 YOLOv8m-P2 方向的内部模型。YOLOv8m-P2 的关键结构在 `models/yolov8m-p2-fire-smoke-3cls.yaml` 中定义，保留 P2/4 高分辨率检测头，并在 P2、P3、P4、P5 四个尺度上输出检测结果，用于缓解远景小火苗和弱烟雾在深层下采样中被削弱的问题。

Stage6 阶段围绕 hard negative 和误报抑制展开。D5 使用 S5 权重作为起点，将 `verified_empty_fp_round01` 和 `verified_empty_fp_consensus_round02` 等空场景与困难负样本加入训练，并重复 hard negative 目录 4 次；训练输入尺寸为 `imgsz=960`，冻结前 10 层，使用 AdamW 和较低学习率 `lr0=0.00003`。D6 从 D5 出发将输入尺寸提高到 1280，但结果明显退化，因此作为高分辨率负向消融。D7 使用与 D5 相同的配置，仅将随机种子改为 1，用于验证 D5 策略的可重复性。

Stage7 阶段训练官方 YOLOv8n/s/m 外部基线，统一使用 `datasets/stage4_full_tile_sensors3/data.yaml`，输入尺寸为 960，训练上限为 100 epoch，patience 为 20，并统一使用 AdamW 和 deterministic seed 0。

当前固定评估协议包括主验证集指标、public-hard 困难场景指标、false_alarm_v2 空场景误报指标和多阈值扫描。主验证集读取 Ultralytics `results.csv` 中的 Precision、Recall、mAP50 和 mAP50-95。public-hard holdout 包含 500 张图像和 1813 个目标，用于评价困难场景召回、tiny recall、类别召回和 FP/image。false_alarm_v2 包含 1000 张 verified-empty UAV/背景图像，用于评价 FP/1000 和按类别拆分的误报。

## 3. 主要实验指标

| 模型 | 训练轮数 | 最优轮次 | Precision | Recall | mAP50 | mAP50-95 | public-hard recall@0.20 | tiny recall@0.20 | FP/image@0.20 | false_alarm_v2 FP/1000@0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage6 D5 / D7 | 8 | 5 | 0.83145 | 0.78723 | 0.83146 | 0.57379 | 0.87038 | 0.71973 | 0.520 | 5 |
| Stage6 D6 高分辨率 | 4 | 1 | 0.78868 | 0.77455 | 0.81097 | 0.48266 | 0.87148 | 0.71525 | 0.530 | 6 |
| YOLOv8n baseline | 100 | 87 | 0.81992 | 0.76836 | 0.83330 | 0.56626 | 0.86873 | 0.71973 | 0.646 | 20 |
| YOLOv8s baseline | 70 | 50 | 0.82792 | 0.78093 | 0.84047 | 0.57865 | 0.87424 | 0.72646 | 0.570 | 14 |
| YOLOv8m baseline | 66 | 46 | 0.83413 | 0.79222 | 0.84601 | 0.58055 | 0.87534 | 0.71749 | 0.548 | 11 |

从主验证集看，D5/D7 不是 mAP 最高模型。YOLOv8m baseline 的 mAP50-95 为 0.58055，高于 D5/D7 的 0.57379。因此论文不能写成“本文方法在常规 mAP 上全面超过所有基线”。

从部署误报指标看，D5/D7 具有明显优势。在 false_alarm_v2 的 1000 张 verified-empty 图像上，conf=0.20 时 D5/D7 为 5 FP/1000，YOLOv8n、YOLOv8s 和 YOLOv8m 分别为 20、14 和 11 FP/1000。相对 YOLOv8m，D5/D7 的空场景误报数量下降约 54.5%。相对 YOLOv8s，下降约 64.3%。相对 YOLOv8n，下降 75.0%。

在 public-hard 困难场景中，D5/D7 的 FP/image 为 0.520，低于 YOLOv8n 的 0.646、YOLOv8s 的 0.570 和 YOLOv8m 的 0.548。YOLOv8s 的 tiny recall 略高于 D5/D7，但其 false_alarm_v2 误报也更高。这说明当前模型的优势更接近“低误报平衡点”，而不是单项小目标召回最优。

## 4. 当前对论文的支撑

当前实验可以支撑一条更完整的论文主线：面向无人机早期火灾预警的小目标火焰与烟雾检测方法。该主线包含结构、数据、训练和评估四个层面，而不是单纯的误报控制。

已经有实验证据支撑的内容包括：YOLOv8m-P2 结构保留 P2 高分辨率检测分支，适合写入方法章节；full + tile 的训练策略和 960 尺度切片设计可作为数据处理与训练策略的一部分；fire/other/smoke 三类别建模以及 verified-empty hard negative 校准在误报控制上有明确实证结果；D5/D7 两组 seed 结果一致，可以支撑当前训练策略具有可重复性；D6 高分辨率负向消融说明单纯增大输入尺寸不是稳定改进方向。

暂时不能写成已验证结论的内容包括：本文方法在所有外部基线上 mAP 最优；模型已经具备完整效率优势；AP_small/AP_medium/AP_large 已经完成；视频级 FP/min、首次检测时间和时序稳定性已经充分验证；RGB-T 双流模型、注意力模块、新损失函数或 Soft-NMS/WBF 等算法模块已经完成并带来提升。

## 5. 修订后的论文定位

当前更合适的论文定位是：

面向无人机早期火灾预警场景，针对远景小火苗和弱烟雾易漏检、复杂背景干扰易误报、常规验证集指标难以反映部署风险等问题，构建一种基于 YOLOv8m-P2 的火焰与烟雾小目标检测方法。该方法通过 P2 高分辨率检测分支增强小目标特征保留，通过 full + tile 数据组织提升大场景小目标学习能力，通过 fire/other/smoke 三类别建模和 verified-empty hard negative 校准抑制误报，并建立包含主验证集、困难场景集、空场景误报集和阈值扫描的综合评估协议。

这个定位比“部署感知误报控制”更完整，也更符合投稿论文需要的闭环：先提出无人机场景中的小目标与误报痛点，再指出现有研究对部署误报和固定空场景评估关注不足，再给出结构与训练策略改进，最后用实验指标证明改进效果。

## 6. 下一步计划

优先任务不是继续盲目微调，而是补齐论文闭环证据。

第一，需要建立最终主对比表，将 D5/D7、V4、S5、No-P2、YOLOv8n、YOLOv8s 和 YOLOv8m 放在同一张表中，统一报告 Precision、Recall、mAP50、mAP50-95、tiny recall、FP/image 和 FP/1000。

第二，需要补齐 P2 消融和 full+tile 消融。P2 是当前论文最明确的结构创新，必须用 No-P2 和 P2 的直接对照证明其作用。full+tile 是当前数据策略创新，也需要有 full-only 对照。

第三，需要采集效率指标，包括 Params、GFLOPs、模型大小、FPS 和 latency。没有这些指标，论文难以证明方法适合无人机或边缘部署。

第四，需要补充视频级评价，包括 FP/min、首次检测时间、漏检事件数和时序闪烁率。早期预警论文如果只做单帧图像评价，支撑力不足。

第五，如果要进一步强化“算法创新”，应选择一个单独可消融的模块方向，例如动态采样或特征对齐颈部、注意力引导特征融合、NWD/WIoU 小框定位损失、Soft-NMS/WBF 多火点后处理。该模块必须单独训练和消融，不能与 P2、hard negative 和切片策略混在一起，否则无法解释提升来源。

## 7. 当前结论

当前实验已经具备写论文初稿的基础，但论文主张需要修正为“无人机小目标火焰与烟雾检测方法及其误报抑制评估”，而不是单独强调误报控制。D5/D7 的结果可以作为现阶段主模型证据，但论文若要投稿，还需要补齐结构消融、数据策略消融、效率指标和视频级评价。
