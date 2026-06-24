# 当前实验对论文的支撑与下一轮优化计划

生成日期：2026-06-16

## 1. 论文主叙事建议

当前实验最适合支撑的论文叙事不是“提出一个全面显著优于 YOLOv8 的新模型”，而是：

> 面向无人机远距离早期火灾预警，系统分析火焰/烟雾小目标检测中的数据稀缺、尺度退化和误检问题，并通过高分辨率 UAV 数据扩展、full+tile 训练、P2 小目标检测头和 hard holdout 验证，建立一个可复现的 YOLOv8m-P2 检测框架。

这个叙事更稳，因为我们已有证据显示：

- 数据策略是主要增益来源；
- P2 对整体 mAP 的提升有限，但对 tiny fire 召回有明确收益；
- 继续 fine-tune 没有带来泛化提升，说明当前瓶颈更可能在数据质量、类别混淆、负样本和小目标推理策略，而不是训练轮数；
- 当前模型在 public hard holdout 上达到较高 mAP50-95，但 fire tiny/small 仍是薄弱环节。

## 2. 现有实验支撑点

### 2.1 数据策略支撑

Stage4 将数据从 Sensors 3cls 扩展到 Sensors + FASDD_UAV + D-Fire，并采用 full+tile 训练策略。审计结果显示：

| Split | Images | Fire instances | Smoke instances | Other instances |
|---|---:|---:|---:|---:|
| train | 75,744 | 90,385 | 70,799 | 15,542 |
| val | 38,306 | 47,930 | 33,343 | 5,897 |
| test | 5,264 | 5,494 | 5,136 | 2,401 |

这一组数据可以支撑论文中的方法贡献：

- 引入 UAV/公开火焰烟雾数据增强目标域覆盖；
- 使用 full+tile 训练解决高分辨率大场景与小目标尺度之间的矛盾；
- 保留 empty negative 与 other 类用于误报抑制。

允许声明边界：

- 可以声明“数据扩展和切片训练策略提升了验证集 mAP50-95”；
- 不宜声明“数据策略完全解决了误检问题”，因为 FASDD_UAV 标注仍存在 smoke/fire 混淆风险。

### 2.2 多阶段训练支撑

| Stage | Model | Dataset | Best Epoch | Val mAP50 | Val mAP50-95 | Holdout mAP50 | Holdout mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| R1 | YOLOv8m | Sensors 3cls | 100 | 0.8456 | 0.5736 | - | - |
| S4 E20 | YOLOv8m-P2 | Full+Tile+FASDD+D-Fire | 20 | 0.84231 | 0.58052 | 0.85677 | 0.63861 |
| S4 V4 | YOLOv8m-P2 | Full+Tile+FASDD+D-Fire | 16 | 0.84371 | 0.58371 | 0.92376 | 0.68120 |
| S5 | YOLOv8m-P2 | V4 fine-tune | 12 | 0.83829 | 0.57902 | 0.92841 | 0.68532 |
| Abl | YOLOv8m No-P2 | Full+Tile+FASDD+D-Fire | 44 | 0.84082 | 0.58018 | 0.91585 | 0.67743 |

论文支撑点：

- V4 是综合最佳主模型：val 指标最高，holdout 也接近 S5；
- S5 证明继续 fine-tune 不是主要提升方向；
- No-P2 消融提供了 P2 贡献的公平对照。

允许声明边界：

- V4 相比 R1 的 val mAP50-95 提升约 +0.0101；
- V4 相比 No-P2 的 val mAP50-95 只提升约 +0.0035；
- S5 holdout 略高，但 val 退化，因此不适合作为主模型。

### 2.3 P2 小目标支撑

Public hard holdout 包含 500 张未参与训练的 FASDD_UAV test 图像，其中 fire 实例 1,278 个，smoke 实例 535 个。fire 中 tiny+small 占 977/1278，约 76.5%，符合无人机远距离小火焰场景。

Per-size 分析结果显示：

| Model | Tiny Fire AP50 | Tiny Fire AP50-95 | Tiny Fire Recall50 |
|---|---:|---:|---:|
| V4_P2 | 0.20134 | 0.07904 | 0.91014 |
| NoP2_Abl | 0.19367 | 0.07276 | 0.83641 |
| Delta | +0.00767 | +0.00627 | +0.07373 |

论文支撑点：

- P2 对整体 mAP 的贡献弱，但对 tiny fire Recall50 有实际价值；
- 这可以成为“P2 主要改善早期小火焰召回”的核心消融结论；
- 但 small fire AP 基本持平，large fire 甚至 No-P2 更高，因此不能把 P2 写成全尺度全面增强模块。

建议论文措辞：

> The P2 detection head yields only a marginal gain in overall mAP, but improves tiny-fire recall on the UAV hard holdout, suggesting that high-resolution shallow features are primarily beneficial for early-stage small fire instances rather than for all object scales.

### 2.4 泛化与部署支撑

V4 在 holdout 上达到 mAP50=0.92376、mAP50-95=0.68120，明显高于 val mAP50-95=0.58371。这个结果支持“public UAV hard holdout 上模型表现较好”，但也提示：

- holdout 可能比 val 更偏向 FASDD_UAV，难度结构不同；
- holdout 无 other GT，不能充分评估误报抑制；
- 需要加入 negative-only / hard-background 测试集，才能支撑“低误报早期预警”。

## 3. 对照高水平论文后的实验范式

高质量 YOLO 小目标/火焰烟雾检测论文通常具备以下实验层：

| 实验层 | 高水平论文常见做法 | 我们当前状态 | 缺口 |
|---|---|---|---|
| Main comparison | 与 YOLOv5/YOLOv8/YOLOv9/YOLOv10/RT-DETR 等多模型比较 | 主要是内部阶段对比 | 缺少外部强基线 |
| Ablation | 对每个模块逐一消融 | 已有 No-P2；数据策略为阶段式对比 | 缺少 tile、FASDD、D-Fire、loss、NMS 消融 |
| Per-size | 报告 small/medium/large 或 tiny/small 指标 | 已补 per-size | 需要加入标准 COCO small/medium/large 口径或解释自定义口径 |
| Efficiency | Params、FLOPs、FPS、延迟、设备 | 部分有 GPU 推理记录 | 需要统一脚本和表格 |
| Robustness | 光照、烟雾、复杂背景、空负样本 | 有误检分析但未系统量化 | 需要 hard negative benchmark |
| Qualitative | 可视化成功/失败案例 | 部分视频验证 | 需要论文级图组 |
| Statistics | 多 seed 或置信区间 | 当前多为单次训练 | SCI一区/二区建议至少关键实验重复或 bootstrap CI |

文献范式参考：

- SAHI 论文强调高分辨率图像中小目标像素少、常规检测器困难，并用 VisDrone/xView 验证切片推理和切片微调的增益。
- SOD-YOLO 类 UAV 小目标论文通常把 P2 小目标层、多尺度特征融合和 NMS 改进作为独立模块，并逐项消融。
- UAV wildfire smoke detection 的 Sensors 论文将 WIoUv3、注意力/特征增强模块、复杂天气场景和 baseline 对比结合起来。
- 近期 fire detection YOLO 论文通常同时报告精度、参数量、GFLOPs 和推理速度，用“准确率-效率权衡”作为重点。

## 4. 当前论文能支撑到什么级别

### 4.1 可以支撑一篇工程应用型 SCI 二区/三区论文的部分

如果定位为“无人机火焰/烟雾检测系统与实验分析”，当前证据已经有基础：

- 数据构建和审计完整；
- 多阶段训练记录完整；
- V4/S5/No-P2 权重与结果已归档；
- hard holdout 与 per-size 分析已补齐；
- 误检根因有初步分析；
- 有部署端视频测试经验。

### 4.2 距离 SCI 一区/强二区的主要差距

若目标是一区或强二区，仅靠当前结果还不稳。主要短板：

1. 创新点偏弱：P2 本身不是新模块，整体 mAP 增益也很小；
2. 外部基线不足：缺少 YOLOv8m、YOLOv5m、YOLOv10/11、RT-DETR 等同数据对比；
3. 消融不够完整：P2 有了，但 tile、数据源、loss、后处理、SAHI 尚未形成表格；
4. 误检评估不足：holdout 没有 other GT，不能支撑低误报；
5. 统计稳健性不足：关键结论基于单次训练；
6. 文献证据未完全转成 evidence-claim map，引用还不能直接进入正文。

## 5. 下一轮训练与实验优化路线

### P0：先补实验，不急着换大模型

1. 统一评估脚本
   - 输入：V4、S5、No-P2、未来新模型；
   - 输出：overall、per-class、per-size、negative FP、FPS、Params、FLOPs；
   - 固定阈值和 NMS 参数，避免不同脚本造成不可比。

2. 补 hard negative benchmark
   - 来源：FASDD 空标签、D-Fire empty、无人机背景、云雾、棕色地表、黄昏高亮区域；
   - 指标：FP/image、FP/min、fire false alarm rate；
   - 这是早期预警论文非常关键的实用指标。

3. 补 SAHI / full+sliced 推理实验
   - Standard 960；
   - SAHI 640 overlap 0.25；
   - SAHI 960 overlap 0.25；
   - Full image + sliced ensemble；
   - 指标重点：tiny fire recall、FP、FPS。

### P1：下一轮训练建议

建议下一次不要只“继续训 V4”，而是开三个有明确假设的实验：

| 实验 | 假设 | 关键改动 | 成功标准 |
|---|---|---|---|
| T1 Focal/Varifocal Loss | 类别混淆和难例导致 FP/FN | focal_gamma 或分类损失重加权 | FP 降低，tiny fire recall 不下降 |
| T2 Corrected Smoke Mapping | FASDD smoke/fire 标注映射影响混淆 | 重新核验 FASDD smoke 类别，修复大烟框误标 | smoke AP 稳定，fire FP 降低 |
| T3 P2 + Soft-NMS / WBF | 密集小火点被 NMS 抑制 | 类别自适应 NMS、Soft-NMS 或 WBF | dense fire recall 提升，FP 可控 |

不建议下一轮一口气加入 BiFormer、WIoU、Focal、Soft-NMS、SAHI fine-tuning、重标注全部模块。这样即使提升，也无法解释是哪一项带来的。

### P2：如果冲 SCI 一区/强二区，需要增加模型创新

可考虑一个“轻量但可解释”的模块组合：

1. P2 小目标头：保留，作为 tiny fire recall 基础；
2. Smoke-aware / fire-aware loss：针对 fire-smoke 混淆；
3. Dynamic tile inference：根据目标尺度或图像分辨率自适应切片；
4. Hard negative mining：把误报帧回流训练；
5. 类别自适应后处理：fire/smoke 使用不同 NMS 和颜色/时序一致性约束。

论文贡献可从“改 YOLO”升级为：

> A data-and-deployment-aware UAV early fire warning framework that combines scale-aware detection, hard-negative mining, and sliced inference for tiny fire and smoke targets.

## 6. 推荐论文图表结构

### Table 1：Dataset Statistics

展示 Sensors、FASDD_UAV、D-Fire、full+tile、holdout 的图像数、实例数、tiny/small 比例、negative 数量。

### Table 2：Main Comparison

至少包含：

- YOLOv8m No-P2；
- YOLOv8m-P2 V4；
- Stage5 fine-tune；
- 后续补 YOLOv5m / YOLOv10m / RT-DETR 或 YOLO11m。

指标：

- Precision、Recall、mAP50、mAP50-95；
- Params、GFLOPs、FPS；
- Holdout mAP50-95。

### Table 3：Ablation Study

建议行：

- baseline Sensors；
- +FASDD/D-Fire；
- +full+tile；
- +P2；
- +Focal/Hard negative；
- +SAHI inference。

### Table 4：Per-size Analysis

重点展示 fire 的 tiny/small/medium/large AP 和 Recall。

### Figure 1：Method Overview

无人机输入图像 → full+tile training → YOLOv8m-P2 → standard/SAHI inference → temporal or post-processing early warning。

### Figure 2：Training Curves

V4、S5、No-P2 的 mAP50-95 与 val loss 曲线，展示 V4 收敛、S5 过拟合。

### Figure 3：Qualitative Results

成功案例、tiny fire、large smoke、误检案例、No-P2 vs P2 对照。

### Figure 4：Failure Analysis

smoke→fire、bright flame→smoke、background→fire 三类错误。

## 7. 结论

当前实验已经足以支持一个扎实的工程型论文雏形：数据扩展、full+tile、P2 消融、hard holdout 和 tiny fire per-size 分析构成了主要证据链。若目标是重点 SCI 一区/二区，需要把实验从“已有训练结果总结”升级为“严谨可复现的研究设计”：补外部强基线、补 hard negative、补 SAHI、补完整消融、补统计稳健性，并将下一轮训练聚焦到明确瓶颈：fire/smoke 混淆、tiny fire recall、误报控制和部署效率。
