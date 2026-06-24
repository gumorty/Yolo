# Stage6 执行日志

日期：2026-06-16  
目标：降低误报、修复火/烟错检和漏检，同时保留 V4-P2 对 tiny fire 的召回优势。

## 1. 已完成动作

### 1.1 FastAPI 测试环境

前端服务已启动：

- URL: http://127.0.0.1:8000/
- 默认模型：`stage4_v4_p2_3cls`
- 已补充模型：`ablation_nop2_yolov8m_3cls`

当前可在前端直接对比：

- V4 YOLOv8m-P2 3cls (Best)
- V5 YOLOv8m-P2 3cls (E120 FT)
- No-P2 YOLOv8m 3cls (Ablation)

### 1.2 数据集盘点

主要训练/评估数据：

| 数据集 | 图片数 | 标签数 | 用途 |
|---|---:|---:|---|
| `stage4_mixed_3cls` | 108422 | 108422 | Stage4 主训练混合集 |
| `stage4_full_tile_sensors3` | 238630 | 238630 | full + tile 小目标增强训练集 |
| `stage4_eval/public_hard_holdout` | 1002 | 1002 | 论文固定 hard holdout |
| `downloads` | 118352 | 118352 | 下载/原始候选数据 |
| `stage4_sources` | 101864 | 101864 | FASDD/D-Fire 等整理来源 |

轻量抽样画像显示：

- `public_hard_holdout` 里 tiny fire 很多，适合评估小火点召回。
- smoke tiny 样本明显偏少，现有烟雾目标以 large smoke 为主。
- Stage4 混合集已经有空标签，但还不是“模型真实误报挖掘”得到的 hard negative。

结论：下一轮最重要的数据动作不是盲目扩大数据量，而是增加“真实误报负样本”“smoke-only/weak smoke”“人工复核后的错检修正”。

### 1.3 新增工具

新增误报/漏检挖掘工具：

- `tools/stage6_mine_hard_cases.py`

功能：

- 从图片数据集挖掘 `false_positive`、`missed_or_incomplete`、`mixed_fp_miss`。
- 从视频中抽取有检测的帧，标记为 `video_detection_needs_review`。
- 保存原图、带框图、crop、`review_manifest.csv`、`review_manifest.jsonl`、`summary.json`。
- 默认跳过 `result_*`、`annotated_*`、`corrected_*`、`final_*` 等派生视频。

新增复核结果导出工具：

- `tools/stage6_review_to_yolo.py`

功能：

- 读取人工编辑后的 `review_manifest.csv`。
- 将 `review_decision=confirmed_empty` 的样本导出成 YOLO 空标签 hard-negative 数据集。
- 输出 `data.yaml`，可被 Stage6 混合配置引用。

### 1.4 新增配置

- `configs/stage6_dataset_mix_template.yaml`
- `configs/stage6_experiment_matrix.yaml`
- `configs/stage6_mining_round01.yaml`

这些文件把 Stage6 分成四层：

1. 基线复评：V4/S5/No-P2。
2. hard-negative 挖掘和复核。
3. Stage6 clean 数据混合。
4. D1/D2 训练与阈值/NMS/slice 扫描。

## 2. Round01 挖掘结果

输入：

- 视频：`D:/Researching/Yolo/Yolo/docs/video`
- 图片：`D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage4_eval/public_hard_holdout/images`
- 过滤后原始视频数量：17
- 每个视频最多保存 30 个候选检测帧
- 图片最多抽取 300 张
- 推理参数：`imgsz=960`，`conf=0.15`，`iou=0.65`，`match_iou=0.50`

输出目录：

| 模型 | 输出目录 | 候选数 | mixed_fp_miss | false_positive | missed_or_incomplete | video_detection_needs_review |
|---|---|---:|---:|---:|---:|---:|
| V4 | `datasets/stage6_mining/round01_v4` | 334 | 87 | 63 | 12 | 172 |
| S5 | `datasets/stage6_mining/round01_s5` | 331 | 91 | 55 | 10 | 175 |
| No-P2 | `datasets/stage6_mining/round01_nop2` | 328 | 87 | 66 | 14 | 160 |

初步判断：

- S5 图片 false positive 略少，但 mixed_fp_miss 更多，不能仅凭 holdout mAP 选择 S5。
- No-P2 false positive 更多，且 tiny fire recall 已知弱于 V4-P2，说明 P2 仍是主线。
- 三模型候选规模接近，说明误报/错检是数据与阈值策略层面的系统问题。

## 3. 人工复核规则

请打开每个目录下的：

- `review_manifest.csv`
- `annotated/`
- `review_images/`
- `crops/`

在 `review_manifest.csv` 的 `review_decision` 列填写：

| 决策值 | 含义 | 后续处理 |
|---|---|---|
| `confirmed_empty` | 确认无火/无烟，是误报背景 | 导出为空标签 hard negative |
| `relabel_fire` | 画面中有漏标/错框火焰 | 后续补 fire bbox |
| `relabel_smoke` | 画面中有漏标/错框烟雾 | 后续补 smoke bbox |
| `relabel_other` | 是应定位的混淆物 | 后续补 other bbox |
| `ignore_uncertain` | 无法确定或画质太差 | 不进入训练 |

导出 hard negative 命令示例：

```powershell
python tools/stage6_review_to_yolo.py `
  --manifest D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_mining/round01_v4/review_manifest.csv `
  --out D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_sources/hard_negative_round01
```

导出后，把 `configs/stage6_dataset_mix_template.yaml` 中的 `stage6_mined_confirmed_empty.enabled` 改为 `true`。

## 4. 训练前数据构建命令

```powershell
python tools/build_stage4_mixed_dataset.py `
  --config D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/configs/stage6_dataset_mix_template.yaml `
  --out D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_mixed_3cls `
  --overwrite
```

构建后运行审计：

```powershell
python tools/dataset_audit.py `
  --data D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_mixed_3cls/data.yaml `
  --out D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/reports/stage6_mixed_3cls_audit.json
```

## 5. D1 快速训练命令

```powershell
python tools/train_yolo_api.py `
  --model D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/models/yolov8m-p2-fire-smoke-3cls.yaml `
  --data D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_mixed_3cls/data.yaml `
  --project D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/runs_stage6 `
  --name stage6_v4p2_hardneg_ft_e50 `
  --pretrained D:/Researching/Yolo/Yolo/models_stage4/v4_best.pt `
  --epochs 50 `
  --batch 16 `
  --imgsz 960 `
  --device 0 `
  --patience 20 `
  --mosaic 0.15 `
  --mixup 0.0 `
  --copy-paste 0.0 `
  --close-mosaic 10 `
  --lr0 0.0008 `
  --lrf 0.01 `
  --weight-decay 0.0005 `
  --amp true `
  --save-period 10
```

## 6. 外部数据补充决策

本轮不建议立刻下载一个很大的新数据集并混入训练。当前已经有 FASDD 和 D-Fire，短板不是数据量不足，而是负样本治理和 smoke/tiny 分布不均。

可以列为 Stage6.5 的候选：

- Boreal Forest Fire：UAV 采集，含 smoke bbox、视频和分割数据，适合补充真实 UAV forest smoke。
- DFS：含 `fire/smoke/other`，其 `other` 类对降低火焰相似物误报有参考价值。
- SKLFS-WildFire Test：包含大量负样本视频，适合做 false positive 测试协议参考。

优先级：

1. 先复核 round01 并训练 D1。
2. 如果 D1 对 false positive 改善不足，再引入 DFS/Boeral Forest Fire。
3. 引入任何新数据前必须先审计标签和类别定义，不能直接混入。

## 7. 文献依据

- SAHI / slicing fine-tuning：小目标在高分辨率图像中会因像素占比低而难检，切片推理和切片微调能提升小目标检测。
- YOLO sliced inference：切片推理和切片 fine-tuning 对 aerial small object 检测有效。
- Wildfire smoke detection system：公开大规模负样本测试集，强调 false positives 是 wildfire detection 的关键问题。
- FASDD：覆盖火、烟和混淆非火/非烟图像，但需要清洗类别映射。
- D-Fire：提供 fire/smoke 和大量 none 样本，适合负样本补充，但非 UAV 主域。
- DFS：使用 `other` 类标注火焰相似干扰物，适合支持我们保留三类任务设计。

## 8. 下一步

1. 使用 `verified_empty` 自动挖掘代替全量人工复核。
2. 将至少两个模型共识误报的空标签样本作为高置信 hard negative。
3. 人工复核只保留 Top100 高价值案例，用于论文失败案例分析和少量补标，不再阻塞训练。
4. 构建 `stage6_mixed_3cls_union`。
5. 运行 D1 快速训练。
6. 用 public hard holdout、false alarm set、视频套件重新评估 V4/S5/No-P2/D1。
7. 如果 D1 达不到门槛，再启动 D2 sliced fine-tuning 和后处理扫描。

## 9. 自动决策替代人工复核

由于 round01 生成的候选案例数量较大，后续流程调整如下：

### 9.1 零人工 hard negative 来源

工具：

- `tools/stage6_mine_verified_empty.py`

原则：

- 只扫描已有 YOLO 空标签图片。
- 如果 V4/S5/No-P2 对空标签图片仍预测出目标，则该图片可自动作为 hard negative。
- `min_models=1` 得到较多样本，适合扩大覆盖。
- `min_models=2` 得到更高置信样本，适合作为 D1 的主 hard-negative 来源。

已完成：

- `datasets/stage6_sources/verified_empty_fp_round01`
- 扫描 3000 张空标签训练图，导出 200 张模型误报 hard negative。
- 已重平衡为 170 train / 30 val。

正在执行：

- `datasets/stage6_sources/verified_empty_fp_consensus_round02`
- 扫描最多 12000 张空标签训练图。
- 只保留至少两个模型共识误报样本。
- 目标最多导出 2000 张。

### 9.2 少量人工复核队列

工具：

- `tools/stage6_triage_review_queue.py`

结果：

- 输入 V4/S5/No-P2 共 993 条候选。
- 合并为 386 个唯一案例。
- 输出 Top100：
  - `datasets/stage6_mining/round01_triage_top100/prioritized_review_queue.csv`

这 100 个案例只用于：

- 检查自动策略是否合理。
- 找论文定性失败案例。
- 少量补 fire/smoke/other 标签。

它们不再是 D1 训练的阻塞条件。

### 9.3 当前训练数据策略

当前 D1 使用：

- `datasets/stage6_mixed_3cls_union/data.yaml`

这个 YAML 不复制大数据集，而是引用：

- Stage4 full/tile 主训练集。
- verified-empty hard negative 数据集。

这样能在不扩大磁盘占用的情况下，把自动挖出的误报负样本并入训练。
## 2026-06-17 本机 3070 D1 训练启动

已在 RTX 3070 8GB 本机上启动 Stage6 D1 训练。该运行用于先验证完整 Stage6 D1 配方，后续仍需要在 4090 服务器上复现实验。

运行目录：

- `runs_stage6_local3070/stage6_d1_local3070_conservative_e20_b2_i960`

训练设置：

- 模型结构：`models/yolov8m-p2-fire-smoke-3cls.yaml`
- 预训练权重：`D:/Researching/Yolo/Yolo/models_stage4/v4_best.pt`
- 数据集：`datasets/stage6_mixed_3cls_union/data.yaml`
- 训练轮数：20
- batch：2
- 图像尺寸：960
- 设备：0
- workers：2
- 冻结层数：10
- AMP：true
- 学习率：`lr0=0.0001`，`lrf=0.2`
- 数据增强：`mosaic=0.05`，`mixup=0.0`，`copy_paste=0.0`，`auto_augment=None`，`erasing=0.0`

启动后状态：

- 进程 PID：`22268`
- 未出现 CUDA OOM。
- 已进入 epoch 1/20。
- 观察速度约为 5.3 it/s。
- 每个 epoch 约 38,006 个 batch。
- 本机预计每个 epoch 约 2 小时，20 个 epoch 约 40 小时。

本机运行定位：

- 该运行不替代最终 4090 双卡结果。
- 它用于验证 Stage6 D1 在全量 union 数据上的训练策略；如果通过固定评估门槛，也可以作为论文辅助证据。

训练后评估门槛：

- `public_hard_holdout`：tiny recall、fire recall、smoke recall、FP/image。
- `uav_false_alarm_v2`：FP/1000 和 FP/image。
- balanced 候选目标：false-alarm v2 的 FP/1000 不高于 S5 基线 5。
- 可接受候选目标：false-alarm v2 的 FP/1000 不高于 V4 基线 9。
## 2026-06-17 D1 本机训练纠错

对本机 Stage6 D1 训练日志进行检查后，发现最初的运行：

- `runs_stage6_local3070/stage6_d1_local3070_conservative_e20_b2_i960`

没有完成 20 个 epoch。该运行完成 2 个 epoch 后被主动停止。

停止原因：

- 虽然命令中设置了 `lr0=0.0001`，但 `results.csv` 显示实际学习率升到 `0.00999974` 和 `0.01919970`。
- 日志显示该运行使用 `optimizer=auto`，并保留默认 `warmup_bias_lr=0.1`。
- 前 2 个 epoch 的 mAP50-95 为 0.56601 和 0.56745，低于 V4 基线的 0.58371。
- 因此该运行不适合作为论文主结果，只保留为失败先导实验。

已执行修正：

- 修改 `tools/train_yolo_api.py`，新增 `--optimizer`、`--warmup-bias-lr`、`--momentum` 参数。
- 更新 `configs/stage6_experiment_matrix.yaml`，正式 D1 命令显式使用 `optimizer=AdamW`、`warmup_epochs=0.0`、`warmup_bias_lr=0.0001`。
- 更新 `tools/stage6_eval_d1_local3070.ps1`，训练完成后评估 corrected D1 权重。

新的有效本机 D1 运行：

- `runs_stage6_local3070/stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4`

当前状态：

- 进程 PID：`26572`
- 已进入 epoch 1/20。
- 未出现 CUDA OOM。
- 日志确认 `optimizer=AdamW`、`lr0=0.0001`、`warmup_epochs=0.0`、`warmup_bias_lr=0.0001` 生效。
- 该 corrected run 才作为当前本机 Stage6 D1 有效训练。

epoch 1 更新：

- corrected run 已完成 epoch 1，并进入 epoch 2。
- epoch 1 指标：Precision=0.81682，Recall=0.78657，mAP50=0.83402，mAP50-95=0.56437。
- 实际学习率：`lr/pg0=0.0001`，`lr/pg1=0.0001`，`lr/pg2=0.0001`。
- 学习率修正有效，但 epoch 1 的 mAP50-95 低于 V4 val mAP50-95=0.58371。
- 决策：继续观察到 epoch 3；若 epoch 3 仍明显低于 0.57，则停止本机长训并转向服务器正式训练。
