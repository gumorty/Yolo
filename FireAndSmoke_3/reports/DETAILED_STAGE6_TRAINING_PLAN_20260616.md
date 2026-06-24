# Stage6 训练与论文实验补强方案

日期：2026-06-16  
项目：UAV 视角火焰/烟雾小目标检测  
当前关键模型：V4 YOLOv8m-P2、S5 fine-tune、No-P2 ablation

## 1. 当前实验能支撑什么论文结论

当前实验已经能支撑一个清晰的工程型论点：在 UAV 远距离、小目标火焰场景下，引入 P2 小目标检测层的 YOLOv8m 变体可以明显提升 tiny fire 的召回能力，同时保持可部署的推理效率。

已经有力的证据：

| 证据 | 本项目结果 | 可写入论文的作用 |
|---|---:|---|
| V4 主模型 | Val mAP50 0.84371，mAP50-95 0.58371 | 作为当前主模型性能 |
| S5 fine-tune | Holdout mAP50 0.92841，mAP50-95 0.68532 | 说明继续微调有轻微外部集收益，但不足以宣称稳定提升 |
| No-P2 对照 | Val mAP50 0.84082，mAP50-95 0.58018 | 作为结构消融基线 |
| tiny fire Recall50 | V4-P2 0.91014，No-P2 0.83641 | 最强论文支撑点：P2 对 tiny fire recall 有明显贡献 |

需要谨慎表述的地方：

- 不能只用 mAP 宣称模型“全面优于”No-P2，因为整体 mAP 增益较小。
- 可以重点写成：P2 对 UAV 小火点早期发现更有价值，尤其体现在 tiny fire Recall50。
- S5 目前不能作为最终主模型，只能作为“继续微调收益有限/可能过拟合”的实验观察。

## 2. 当前不足与可能原因

### 2.1 误报严重

可能原因：

- 训练集中 hard negative 不足，模型没有充分见过太阳反光、灯光、红色屋顶、土壤、云雾、尘土、河面反射、低照度噪声等易混背景。
- 只看 mAP 时，空场景误报不会被充分惩罚；论文实验缺少 FP/image、FP/minute 这类早期预警系统更关心的指标。
- smoke 阈值当前较低，演示系统默认 smoke_conf=0.10，容易召回更多烟雾，也可能扩大误报。

### 2.2 检测错误：fire/smoke/other 混淆

可能原因：

- FASDD、Sensors、D-Fire 等数据源类别定义不完全一致，合并时若只做机械映射，会把“光源/火焰边缘/浓烟遮挡火焰”带入噪声。
- UAV 远景中 smoke 是低对比、半透明、边界模糊目标；fire 是高亮、纹理不稳定目标，两类在小尺度下很容易被模型按颜色和局部纹理误判。

### 2.3 检测不完整

可能原因：

- 大烟雾区域或边界弥散目标用 bbox 标注时本身有主观性，训练集标注不一致会导致预测框偏小或漏掉边缘。
- full image 推理对小目标仍然吃亏；slice/full_sliced 推理和切片微调还没有系统纳入训练闭环。
- NMS 可能压掉相邻小火点，或者在 full+sliced 融合时造成重复框/漏框。

### 2.4 论文级实验不足

目前还缺：

- 专门的 hard-negative benchmark。
- 外部数据集泛化测试。
- 多随机种子或 bootstrap 置信区间。
- 强基线：YOLOv8m/no-P2、YOLOv8m-P2、YOLOv8s/l、YOLOv10/YOLOv11、RT-DETR 等统一协议比较。
- 完整消融：P2、数据清洗、hard negatives、SAHI/sliced fine-tuning、阈值策略、后处理策略分别带来什么变化。

## 3. 文献启发与决策

### 3.1 小目标检测：切片推理与切片微调

SAHI 提供了一个很适合本项目的路线：对高分辨率图像做切片推理，并进一步用切片样本参与 fine-tuning。相关工作在 aerial/surveillance 小目标上报告了明显 AP 提升。YOLO sliced inference 研究也显示，切片推理和切片微调结合通常比只做 full image 推理更适合小目标。

决策：

- 下一轮必须把 `full`、`sliced`、`full_sliced` 三种模式纳入正式实验。
- 训练集不只保留原图，还要构建 sliced fine-tuning 子集。
- 论文中把“P2 小目标头 + 切片推理/切片微调”作为一条完整小目标增强链路。

### 3.2 UAV 火/烟检测：复杂背景和实时部署

UAV wildfire smoke/fire 相关论文通常不只报告 mAP，还强调实时性、边缘部署、复杂林区背景、早期发现能力。本项目演示系统已经具备图像/视频检测和参数调节，适合把“工程可用性”写实。

决策：

- 增加视频级指标：FP/min、稳定检测帧数、首次检测时间、漏检片段数。
- 增加部署指标：FPS、模型参数量、FLOPs、显存占用、不同 imgsz 下的速度-精度曲线。

### 3.3 FASDD 与异构数据

FASDD 等大规模异构火烟数据可以提高泛化，但它们跨场景、跨尺度、跨标注风格。直接合并能扩大数量，却也可能引入类别和 bbox 噪声。

决策：

- FASDD 只能作为“清洗后扩展集”，不能盲目全量加入。
- 先抽样审计标签，再建立类别映射和排除规则。
- 对烟雾边界、大框火焰、明显非 UAV 近景样本做单独标记，避免污染 UAV 小目标主任务。

## 4. Stage6 数据集方案

### 4.1 数据版本命名

建议建立：

- `data_stage6_clean_3cls`：清洗后的主训练集。
- `data_stage6_sliced_3cls`：切片微调集。
- `uav_hard_test_v2`：论文级固定测试集，不参与训练。
- `uav_false_alarm_v1`：纯负样本/易误报测试集。

### 4.2 类别定义

保持三类：

- `fire`：可见火焰、火点、燃烧高亮区域。
- `smoke`：烟柱、烟雾团、低对比扩散烟。
- `other`：与火烟相关但不应报警的混淆物，或局部非目标显著物。

注意：如果图片完全无目标，应使用空 label，而不是强行标 `other`。`other` 应只用于确有需要模型学习定位的混淆目标。

### 4.3 数据组成建议

| 子集 | 建议比例 | 目的 |
|---|---:|---|
| 当前 Sensors/UAV 主集 | 35%-45% | 保持 UAV 主域 |
| 当前 V4 使用的 FASDD 清洗样本 | 20%-30% | 增强场景多样性 |
| smoke-only 与 early smoke 样本 | 10%-15% | 修复烟雾低召回和类别混淆 |
| tiny fire 密集样本 | 10%-15% | 保持 P2 小火点优势 |
| hard negative/empty | 15%-25% | 降低误报 |
| 视频抽帧样本 | 5%-10% | 提升连续帧稳定性 |

### 4.4 hard negative 采样清单

必须加入的负样本：

- 太阳、灯光、车灯、反光玻璃、金属反射、水面反光。
- 红色/橙色屋顶、土地、秋季植被、施工区域。
- 云、雾、尘土、水汽、低云、雾霾。
- 夜间噪声、压缩噪声、运动模糊。
- 无人机俯视森林、村庄、农田、河流、工业园空场景。

来源：

- 使用当前前端批量测试视频，把误报帧保存出来。
- 对每个误报框裁剪局部图，同时保留原始全图。
- 每次训练后做 error mining，形成 `hard_neg_round_01/02/03`。

### 4.5 标签审计

抽样审计规则：

- 每个数据源随机抽 300 张。
- 每类至少抽 100 个 bbox。
- 对 tiny fire、smoke-only、empty、hard negative 单独抽样。
- 标注问题分为：错类、漏标、框过大、框过小、边界不一致、应为空标签。

通过条件：

- 错类率低于 3%。
- 漏标率低于 5%。
- hard negative 中明显火/烟漏标样本必须剔除或补标。

## 5. Stage6 训练实验矩阵

### 5.1 基线复评

先不训练，复评现有模型：

| 编号 | 模型 | 目的 |
|---|---|---|
| B0 | V4 YOLOv8m-P2 | 当前主基线 |
| B1 | S5 fine-tune | 检验微调收益是否稳定 |
| B2 | No-P2 YOLOv8m | 结构消融 |
| B3 | legacy 2cls/3cls | 作为历史对照 |

统一评估集：

- val original
- public holdout
- `uav_hard_test_v2`
- `uav_false_alarm_v1`
- representative videos

### 5.2 数据改进实验

| 编号 | 改动 | 要回答的问题 |
|---|---|---|
| D1 | V4 架构 + Stage6 clean 数据 | 数据清洗本身能不能降误报 |
| D2 | D1 + hard negatives | hard negatives 对 FP/image 的贡献 |
| D3 | D2 + smoke-only 增强 | smoke 召回和 fire/smoke 混淆是否改善 |
| D4 | D3 + sliced fine-tuning | 小目标召回是否进一步提升 |

### 5.3 结构与训练策略实验

优先级从高到低：

| 编号 | 方法 | 风险 | 建议 |
|---|---|---|---|
| M1 | 保持 V4 P2 架构，仅换数据 | 低 | 第一优先级 |
| M2 | P2 + BiFPN/轻量注意力 EMA/CBAM | 中 | 数据实验稳定后再做 |
| M3 | NWD/WIoU/小框友好回归损失 | 中 | 若 tiny AP50-95 不升再做 |
| M4 | Soft-NMS/WBF/class-adaptive NMS | 低 | 可先作为推理后处理消融 |
| M5 | YOLOv10/YOLOv11/RT-DETR 外部强基线 | 中 | 论文主比较必须补 |

不建议一开始就大改网络。当前证据表明最大短板是误报和数据/评估协议，不是主干不够复杂。先用数据和评估把问题打透，再决定是否引入新模块。

## 6. 推荐下一次训练配置

### 6.1 快速验证训练

名称：`stage6_v4p2_hardneg_ft_e50`

- 初始权重：V4 best.pt。
- imgsz：960。
- epochs：50。
- batch：按显存设定，优先 16；不足则 8 并累积梯度。
- optimizer：SGD 或 AdamW 选当前仓库已有稳定配置。
- lr0：0.0005-0.001。
- mosaic：0.1-0.2。
- close_mosaic：10。
- mixup/copy_paste：关闭或极低，避免烟雾边界被合成破坏。
- 早停：patience 20。
- 保存：best、last、每 10 epoch checkpoint。

用途：快速判断 hard negatives + 标签清洗是否有效。

### 6.2 论文最终候选训练

名称：`stage6_v4p2_clean_sliced_e100`

- 初始权重：COCO pretrained YOLOv8m-P2 或 V4 best 双路线都跑一次。
- imgsz：960，另补 1280 消融。
- epochs：100。
- 数据：Stage6 clean + hard negatives + sliced fine-tuning samples。
- 评估：每 10 epoch 在 val、holdout、hard-negative 上保存预测。
- 选择标准：不只看 val mAP，以 hard-negative FP/image 和 tiny fire Recall50 共同决定 best。

### 6.3 推理参数实验

必须扫描：

- conf：0.10、0.15、0.20、0.25、0.30、0.40。
- smoke_conf：0.10、0.15、0.20、0.25、0.30。
- iou：0.45、0.55、0.65、0.75。
- mode：full、sliced、full_sliced。
- slice_size：512、640、768。
- overlap：0.20、0.25、0.30。

论文中建议给出两个工作点：

- `Recall-first`：早期预警优先，低漏检，可接受少量误报。
- `Balanced`：面向部署，控制误报。

## 7. 论文实验表格设计

### 7.1 主结果表

字段：

- Model
- Backbone/Head
- P2
- Training data
- mAP50
- mAP50-95
- Fire AP50
- Smoke AP50
- Tiny Fire Recall50
- FP/image on hard-negative
- FPS
- Params
- FLOPs

### 7.2 消融表

消融顺序：

1. Baseline YOLOv8m。
2. + P2。
3. + label cleaning。
4. + hard negatives。
5. + smoke-only balancing。
6. + sliced fine-tuning。
7. + class-adaptive NMS。

### 7.3 鲁棒性表

场景：

- daytime
- dusk/night
- haze/fog
- high altitude tiny fire
- forest background
- urban/industrial background
- negative empty frames

指标：

- Recall
- FP/image
- FP/min
- missed-video segments

## 8. 下一步执行顺序

1. 用已启动的前端对 V4、S5、No-P2 做图像和视频误报采样。
2. 保存误报帧、漏检帧、错类帧，建立 `error_mining_round_01`。
3. 建立 `uav_false_alarm_v1` 和 `uav_hard_test_v2`，固定不参与训练。
4. 对 FASDD 和当前训练集做抽样标签审计。
5. 生成 Stage6 clean 数据 YAML。
6. 跑 `stage6_v4p2_hardneg_ft_e50` 快速验证。
7. 若 FP/image 下降明显且 tiny fire Recall50 未显著下降，再跑 `stage6_v4p2_clean_sliced_e100`。
8. 做阈值、NMS、slice 参数扫描。
9. 补齐 YOLOv10/YOLOv11/RT-DETR 等强基线。
10. 生成论文表格、曲线、误报/漏检可视化图。

## 9. 进入 SCI 一区/二区所需最低门槛

建议目标：

- hard-negative FP/image 相对 V4 下降至少 30%。
- tiny fire Recall50 不低于 V4 的 0.91014，或下降不超过 2 个百分点同时 FP 显著下降。
- holdout mAP50-95 至少提升 0.01。
- smoke AP 和 smoke recall 明显改善。
- 至少 3 个强基线统一训练/评估。
- 至少 5 个关键消融项。
- 视频级测试至少 5-10 段，报告 FP/min 和首次检测时间。
- 给出失败案例和限制，不回避夜间、雾霾、反光等问题。

## 10. 推荐引用线索

- SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection, arXiv:2202.06934.
- Evaluation of YOLO Models with Sliced Inference for Small Object Detection, arXiv:2203.04799.
- FASDD: An open flame and smoke detection dataset for deep learning in fire detection, DOI: 10.1080/10095020.2024.2347922.
- UAV wildfire smoke/fire detection and improved YOLOv5/YOLOv8 papers for deployment and complex background motivation.
- Recent smoke detection work using contrast/attention modules can support the discussion of smoke low-level texture difficulty, but should not be copied into the method until our data-level fixes are validated.
