# Stage6 到 4090 长训前实验分析与论文支撑方案

生成时间：2026-06-18  
项目主题：无人机视角火焰与烟雾小目标检测  
当前目标：将 Stage6 的本机先导实验扩展为可支撑论文主实验、消融实验和鲁棒性实验的服务器长期训练方案。

## 1. 当前实验已经能支撑什么

当前实验已经形成了三个可写入论文的基础证据：

1. P2 小目标检测结构具有必要性。V4/S5 与 No-P2 的对比显示，保留 P2 分支有助于维持无人机视角下小目标火焰召回，尤其是在 public hard holdout 的 tiny recall 指标上，No-P2 没有表现出优势。
2. 单纯追求 mAP 不足以评价早期火灾检测模型。当前模型在 public hard 和 false-alarm v2 上表现出明显的阈值敏感性，因此论文应同时报告 mAP、tiny recall、FP/image、FP/1000 empty images 和视频级误报。
3. hard negative 训练对误报抑制有效。corrected D1 在 `conf=0.20` 下将 false-alarm v2 的误报压到 `5/1000`，优于 V4 的 `9/1000` 和 No-P2 的 `15/1000`，达到 S5 的水平。

这些证据足够支撑“问题分析”和“鲁棒性消融”，但还不足以支撑“提出一个最终更优模型”的主结论。主结论必须依赖 4090 上的 D2/D3 长训结果。

## 2. corrected D1 的真实定位

corrected D1 的训练曲线如下：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.81682 | 0.78657 | 0.83402 | 0.56437 |
| 2 | 0.81802 | 0.78083 | 0.82674 | 0.56105 |
| 3 | 0.82591 | 0.78052 | 0.83061 | 0.56324 |
| 4 | 0.82499 | 0.77592 | 0.82626 | 0.55967 |

与 V4 的验证集 mAP50-95 `0.58371` 相比，D1 没有恢复主验证指标。epoch 3 低于 `0.57`，epoch 4 进一步下降，因此停止本机训练是正确决策。

但是 D1 的固定阈值评估显示了一个有价值的消融结论：

| 模型 | public hard tiny recall | public hard FP/image | false-alarm v2 FP/1000 |
|---|---:|---:|---:|
| V4 | 0.70852 | 0.542 | 9 |
| S5 | 0.72197 | 0.538 | 5 |
| No-P2 | 0.70628 | 0.554 | 15 |
| corrected D1 | 0.70852 | 0.522 | 5 |

这个结果说明 D1 不是主模型候选，而是 hard-negative 数据策略的有效性证据：它能降低空场景误报，并且没有牺牲 `conf=0.20` 下的 tiny recall。

## 3. 论文主线应如何调整

建议论文主线不要写成“我们只改进 YOLO 后 mAP 更高”，而应写成：

面向无人机早期火灾场景，模型不仅要检测小火点和稀薄烟雾，还要避免把云、雾、强光、地表纹理和空场景误报为火烟。本文围绕小目标召回、空场景误报和部署阈值稳定性构建训练与评估流程，并验证 P2 小目标分支、hard-negative 挖掘和阈值扫描对无人机火烟检测的作用。

这个主线更稳，因为它与我们已有数据完全一致：P2 负责小目标，D1 证明 hard negative 可降低误报，后续 D2/D3 负责把 mAP 和召回恢复到主结果水平。

## 4. 4090 长训前必须修正的问题

### 4.1 Linux 路径问题

原始 `datasets/stage6_mixed_3cls_union/data.yaml` 使用 Windows 绝对路径，不能直接在 Linux 服务器训练。已修复：

- `tools/stage6_train_d2_server4090.sh` 会在服务器运行时自动生成：
  - `generated/stage6_mixed_3cls_union_server.yaml`

该 YAML 使用服务器项目根目录 `$ROOT` 拼接路径，适配 `/home/uav/gu/...` 环境。

### 4.2 D1 训练策略过于保守

D1 使用冻结 backbone 和低学习率，能抑制误报，但 mAP 恢复不足。4090 上不能简单延长 D1，而要采用两阶段或三阶段策略：

- 先冻结适配 hard negative。
- 再解冻全模型恢复 mAP。
- 必要时再加入轻量增强和更高分辨率训练。

### 4.3 不能只看 best.pt 的常规 mAP

每个服务器候选模型必须同时通过三类评估：

- 原始验证集：Precision、Recall、mAP50、mAP50-95、classwise AP。
- public hard holdout：tiny recall、fire recall、smoke recall、FP/image。
- false-alarm v2：FP/1000、FP/image、FP_by_class。

如果只看 mAP，模型可能在论文主指标上好看，但实际无人机场景误报严重。

## 5. 4090 训练矩阵

### D2：主候选，恢复 mAP 的 hard-negative 两阶段训练

目的：保留 D1 的误报收益，同时恢复 mAP。

脚本：

```bash
bash tools/stage6_train_d2_server4090.sh
```

默认策略：

- phase1：冻结 backbone，训练 3 epoch，学习率 `0.0001`。
- phase2：全量解冻，训练 12 epoch，学习率 `0.00006`。
- 优先从 S5 权重启动，找不到 S5 时回退到 V4。
- 使用 `AdamW`，禁止 `optimizer=auto`。
- 使用极弱增强：`mosaic=0.05/0.03`，`mixup=0.0`，`copy_paste=0.0`。

D2 验收门槛：

| 指标 | 最低要求 | 理想目标 |
|---|---:|---:|
| val mAP50-95 | 接近 V4 的 0.58371 | 高于 0.58371 |
| public hard tiny recall | 不低于 0.70852 | 高于 S5 的 0.72197 |
| public hard FP/image | 不高于 0.522 | 明显低于 0.522 |
| false-alarm v2 FP/1000 | 不高于 5 | 低于 5 |
| smoke recall | 不低于 0.92710 | 高于 S5 的 0.92897 |

### D3：高分辨率小目标强化训练

目的：验证 4090 显存是否能通过 `imgsz=1280` 提升小目标召回。

建议命令：

```bash
IMGSZ=1280 BATCH=8 bash tools/stage6_train_d2_server4090.sh
```

D3 只在 D2 通过误报门槛但 tiny recall 仍不足时执行。D3 的风险是推理速度下降和显存压力上升，因此它应该作为小目标强化消融，不一定作为最终部署模型。

### D4：阈值与部署策略实验

目的：给论文提供可部署操作点，而不是只提供单个权重。

每个候选权重都要扫描：

```bash
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d2_server4090_phase2_unfreeze_s5_union_i960/weights/best.pt
```

论文中建议报告三个操作点：

| 操作点 | conf | 用途 |
|---|---:|---|
| recall-first | 0.15 或 0.20 | 早期预警，优先降低漏检 |
| balanced | 0.20 或 0.25 | 常规无人机巡检 |
| conservative alarm | 0.30 或 0.40 | 降低误报，适合自动告警 |

这样可以把误报和召回的权衡讲清楚，符合应用型 SCI 论文的实验习惯。

## 6. 论文还缺的关键实验

### 6.1 外部模型对比

当前已有 V4、S5、No-P2 和 D1，但这还偏内部。论文主表至少应加入：

- YOLOv8s 或 YOLOv8m 官方结构基线。
- YOLOv10 或 YOLOv11 同尺度模型。
- RT-DETR 或 Faster R-CNN 中至少一个非 YOLO 对比，如果训练成本可接受。
- 轻量边缘模型，如 YOLOv8n，用于速度与精度权衡。

外部对比不一定都要超过，但必须说明我们的方法在 UAV 小目标、误报抑制或部署阈值上有什么优势。

### 6.2 消融实验

建议论文消融表：

| 消融项 | 目的 | 当前状态 |
|---|---|---|
| P2 vs No-P2 | 证明小目标分支必要性 | 已有基础结果 |
| V4 vs S5 | 证明微调收益 | 已有基础结果 |
| V4 vs D1 | 证明 hard negative 抑制误报 | 已有结果 |
| D1 vs D2 | 证明两阶段服务器训练恢复 mAP | 待服务器训练 |
| full inference vs sliced inference | 证明切片推理对小目标的贡献 | 待补 |
| 960 vs 1280 | 证明分辨率对 tiny recall 的影响 | 待补 |

### 6.3 视频级实验

仅图片检测不足以支撑无人机应用论文。视频测试至少需要四个指标：

- FP/min：每分钟误报数量。
- first detection time：火焰或烟雾首次被检测到的时间。
- missed event count：有火烟但整段漏检的事件数量。
- temporal flicker：同一目标检测结果闪烁次数。

当前视频数据源：

- `D:/Researching/Yolo/Yolo/docs`

服务器模型回传后，应在本地 FastAPI 或批处理脚本中跑视频评估，挑选典型成功与失败案例作为论文图。

## 7. 与相关工作的对齐

近期火烟检测和小目标检测论文通常不会只给一个 mAP 表，而会同时强调：

- UAV 场景下烟雾尺度小、形态弱、背景干扰强。
- 小目标检测需要切片推理或高分辨率训练，例如 SAHI 提出 slicing aided inference/fine-tuning 用于提升远距离小目标检测。
- 火烟数据集需要包含非火烟干扰样本，例如 D-Fire 和 FASDD 都包含非火/非烟或复杂背景样本。
- 应用型火灾检测论文常报告速度、参数量、FLOPs、FPS 和误报案例。

因此我们的论文实验应避免只写“mAP 提升”，而要写成“早期预警场景下的召回-误报-效率三目标平衡”。

## 8. 服务器训练后的判定规则

服务器训练完成后，按以下规则判定模型位置：

1. 如果 D2 同时满足 `val mAP50-95 >= 0.58371`、`false-alarm v2 <= 5/1000`、`public hard tiny recall >= 0.72197`，则 D2 可以作为论文主模型。
2. 如果 D2 的误报优秀但 mAP 低于 V4，则 D2 只能作为鲁棒性增强模型，主模型仍用 S5 或 V4。
3. 如果 D3 提升 tiny recall 但速度明显下降，则 D3 写成高精度模式，不作为默认部署模式。
4. 如果阈值扫描显示 `conf=0.20` 与 `conf=0.25` 存在明显取舍，则论文中同时报告 recall-first 和 balanced 两个操作点。
5. 如果所有 Stage6 模型都不能超过 S5，则论文应把贡献聚焦到系统化评估、hard-negative mining 和 UAV 场景鲁棒性，而不是声称模型结构显著优于所有基线。

## 9. 下一步执行清单

1. 将当前项目同步到 4090 服务器，确保 `datasets`、`tools`、`models` 和 `reports` 同步完整。
2. 在服务器项目根目录执行 `bash tools/stage6_train_d2_server4090.sh`。
3. 训练结束后执行 `bash tools/stage6_eval_model_server.sh <best.pt>`。
4. 回传 D2 的 `results.csv`、`args.yaml`、`weights/best.pt`、阈值扫描 CSV 和可视化曲线。
5. 根据 D2 结果决定是否启动 D3 的 `IMGSZ=1280` 训练。
6. 服务器模型稳定后，补视频级评估和外部基线训练。

当前最重要的原则是：4090 长训必须服务于论文证据链，而不是只追求一个更高的训练轮数。D2 的目标是把 D1 已证明的误报抑制能力，与 V4/S5 已有的小目标召回和 mAP 重新合并起来。
