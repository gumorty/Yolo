# 论文实验支撑规划与执行记录（2026-07-02）

## 1. 论文实验目标

当前论文主线调整为：面向无人机早期火灾预警的小目标火焰与烟雾检测方法研究。实验不再只围绕误报控制展开，而是要形成完整闭环：

1. 无人机远景场景下小火焰、弱烟雾目标容易漏检；
2. 复杂背景、云雾、强光、地物纹理容易造成误报；
3. 常规验证集 mAP 难以直接反映早期预警部署风险；
4. 需要从结构、数据、训练、评估和效率五个层面证明方法有效。

## 2. 已有证据

| 证据类型 | 当前状态 | 论文作用 |
|---|---|---|
| YOLOv8m-P2 结构 | 已有 P2/P3/P4/P5 四尺度检测结构 | 支撑小目标检测结构改进 |
| Stage4 full+tile 数据策略 | 已完成 V4/S5 训练 | 支撑大图小目标学习策略 |
| Stage6 hard negative 校准 | 已完成 D5/D7 | 支撑复杂背景误报抑制 |
| YOLOv8n/s/m 外部基线 | 已完成 Stage7 | 支撑与官方 YOLO 系列对比 |
| false_alarm_v2 固定评估 | 已完成 D5/D7 与 YOLOv8n/s/m | 支撑部署风险评价 |

## 3. 仍需补齐的关键实验

### 3.1 P2 结构消融

目标：证明 P2 高分辨率检测头对无人机小目标火焰/烟雾检测有贡献。

执行策略：

- 使用已有 No-P2 YOLOv8m 对照权重；
- 与 V4、S5、D7 在同一 public-hard 与 false_alarm_v2 评估集上扫描阈值；
- 重点比较 tiny recall、public-hard recall、FP/image 与 FP/1000。

执行状态：2026-07-02 已启动并完成已有模型统一阈值扫描。

### 3.2 Full+Tile 数据策略消融

目标：证明全图+切片训练比只使用 full image 混合训练更适合无人机大场景小目标。

执行策略：

- 新训练 full-only P2 scout 对照；
- 模型结构：YOLOv8m-P2；
- 初始化：`sensors_yolov8m_3cls_100_best.pt`；
- 数据：`stage4_mixed_3cls/data.yaml`；
- 对照：已有 `stage4_scout_p2_fasdd_dfire_tile_e20`；
- 指标：mAP50、mAP50-95、public-hard recall、tiny recall、FP/image、false_alarm_v2 FP/1000。

执行状态：2026-07-02 已在服务器启动 `paper_full_only_p2_stage4_mixed_i960_e20_seed0`。

### 3.3 效率与部署指标

目标：证明方法具备无人机/边缘预警可部署性。

执行策略：

- 统计模型权重大小、参数量、FLOPs；
- 在固定 public-hard 子集上评估平均延迟、P50/P95 延迟和 FPS；
- 对比 No-P2、V4、S5、D7、YOLOv8n/s/m。

执行状态：2026-07-02 已启动 `paper_model_efficiency_20260702.csv` 生成任务。

### 3.4 视频级预警评估

目标：补齐单帧 mAP 之外的早期预警证据。

计划指标：

- FP/min；
- 首次检测时间；
- 漏检事件数；
- 时序闪烁率；
- 连续帧确认后的稳定报警率。

执行状态：待 full-only scout 与效率任务稳定后执行。优先选取已有无人机/烟雾测试视频，不再只做图像级评估。

### 3.5 可选算法模块强化

目标：如果当前 P2 + full/tile + hard negative 的创新强度仍不足，再引入一个可独立消融的算法模块。

候选方向：

- NWD/WIoU 小框定位损失；
- 注意力引导的跨尺度特征融合；
- Soft-NMS/WBF 多火点后处理；
- DySample 或轻量化上采样模块。

执行原则：一次只引入一个模块，并和 P2、数据策略、hard negative 分开消融。

## 4. 2026-07-02 已启动服务器任务

| 任务 | PID | 输出 |
|---|---:|---|
| 已有模型统一阈值扫描 | 1190345 | `/home/uav/gu/projects/FireAndSmoke_3/reports/stage6_threshold_scan_paper_*.csv` |
| full-only P2 scout 训练 | 1190460 | `/home/uav/gu/projects/FireAndSmoke_3/runs_paper_ablation/paper_full_only_p2_stage4_mixed_i960_e20_seed0` |
| 模型效率评估 | 1191272 | `/home/uav/gu/projects/FireAndSmoke_3/reports/paper_model_efficiency_20260702.csv` |

## 5. 第一批新增评估结果

在 `conf=0.20` 下，已有模型统一评估得到：

| 模型 | public-hard recall | tiny recall | FP/image | false_alarm_v2 FP/1000 |
|---|---:|---:|---:|---:|
| No-P2 YOLOv8m | 0.86486 | 0.70628 | 0.554 | 15 |
| V4 P2 | 0.86486 | 0.70852 | 0.542 | 9 |
| S5 P2 | 0.87148 | 0.72197 | 0.540 | 5 |

初步观察：P2/full+tile 系列在 tiny recall 与误报控制上相对 No-P2 更稳；S5 在 false_alarm_v2 上将 FP/1000 从 No-P2 的 15 降至 5，同时 public-hard recall 提升到 0.87148。这组结果可以支撑“结构与训练策略共同改善小目标召回和复杂背景误报”的主张，但仍需要结合 D7、YOLOv8n/s/m 和 full-only scout 的最终结果形成主表。

## 6. 第一批效率评估结果

效率评估在 public-hard 固定 200 张图像子集上运行，输入尺寸统一为 `imgsz=960`。结果如下：

| 模型 | Params | GFLOPs@960 | 权重大小（MB） | 平均延迟（ms） | 平均 FPS |
|---|---:|---:|---:|---:|---:|
| No-P2 YOLOv8m | 25,858,057 | 177.912 | 49.69 | 26.601 | 37.592 |
| V4 P2 | 25,052,620 | 221.613 | 48.54 | 29.572 | 33.815 |
| S5 P2 | 25,052,620 | 221.613 | 48.54 | 29.634 | 33.745 |
| D7 hard negative P2 | 25,052,620 | 221.613 | 48.54 | 29.685 | 33.687 |
| YOLOv8n baseline | 3,011,433 | 18.442 | 6.03 | 24.023 | 41.626 |
| YOLOv8s baseline | 11,136,761 | 64.465 | 21.54 | 24.382 | 41.013 |
| YOLOv8m baseline | 25,858,057 | 177.912 | 49.69 | 27.147 | 36.837 |

初步观察：P2 系列的计算量高于标准 YOLOv8m，平均 FPS 从 YOLOv8m baseline 的 36.837 降至约 33.7，代价约为 8.5%；但其权重大小略低于标准 YOLOv8m。论文中应如实表述为“P2 分支带来一定计算开销，但仍保持 30 FPS 以上的单图推理速度”，不能写成效率全面优于标准 YOLOv8m。

## 7. 下一步执行顺序

1. 等待 full-only scout 第 1 轮完成，确认 `results.csv` 正常写入；
2. 等待效率评估完成，整理模型大小、FLOPs、FPS、延迟表；
3. full-only scout 完成后，立即对其运行 public-hard 与 false_alarm_v2 阈值扫描；
4. 将 No-P2、V4、S5、D7、YOLOv8n/s/m、full-only 的结果合并成论文主对比表；
5. 启动视频级评估，形成早期预警场景证据；
6. 若主表仍不足以支撑投稿创新，选择一个独立算法模块做 Stage8 消融。
