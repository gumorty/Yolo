# Stage6 自动决策与训练路线更新

日期：2026-06-17  
目标：在不进行大规模人工复核的前提下，降低误报并保持 UAV 小目标火焰召回。

## 1. 人工复核策略调整

原始 round01 复核队列数量过大，不适合逐张人工判断。因此流程已改为：

1. 自动 hard negative：只使用已有空标签图片中被模型误报的样本。
2. 三模型压缩队列：V4/S5/No-P2 的候选合并为 Top100，只用于抽查、失败案例和少量补标。
3. 视频帧不自动当作负样本，因为无真值，避免把真实火/烟误加入负样本。

已完成：

- `datasets/stage6_sources/verified_empty_fp_round01`：200 张自动误报负样本，170 train / 30 val。
- `datasets/stage6_sources/verified_empty_fp_consensus_round02`：113 张至少两个模型共识误报负样本，97 train / 16 val。
- `datasets/stage6_mining/round01_triage_top100`：从 993 条候选压缩为 386 个唯一案例，并输出 Top100。

## 2. 固定误报评估集

已建立：

- `datasets/stage6_eval/uav_false_alarm_v1`

内容：

- 从 public hard holdout 中抽取 100 张空标签图片。
- 不进入训练，只用于报告 FP/image。

基线结果，`conf=0.20, imgsz=960, IoU=0.50`：

| 模型 | FP / 100 images | FP/image | 误报类别 |
|---|---:|---:|---|
| V4 | 2 | 0.02 | smoke=2 |
| S5 | 1 | 0.01 | smoke=1 |
| No-P2 | 3 | 0.03 | smoke=2, fire=1 |
| Scout aggressive | 6 | 0.06 | smoke=6 |
| Scout conservative | 2 | 0.02 | smoke=2 |

结论：

- 当前 V4/S5 在这个固定空标签集上并不差。
- 小样本 aggressive fine-tune 会恶化误报。
- conservative fine-tune 能保持 fixed false-alarm，但不能进一步降低。

## 3. Scout 训练结果

### 3.1 Aggressive Scout

训练：

- `runs_stage6_scout/stage6_scout_hardneg_ft_e5`
- 约 4 epoch，有 hard negatives，但学习率和更新幅度偏大。

结果：

| 指标 | V4 | Aggressive Scout |
|---|---:|---:|
| false-alarm v1 FP/100 | 2 | 6 |
| public hard tiny recall | 0.7085 | 0.6682 |
| public hard fire recall | 0.8396 | 0.8036 |
| public hard smoke recall | 0.9252 | 0.9196 |

决策：

- 该路线失败，不进入正式训练。

### 3.2 Conservative Scout

训练：

- `runs_stage6_scout/stage6_scout_conservative_e3`
- 3 epoch。
- `lr0=0.0001`
- `freeze=10`
- 关闭 mosaic/mixup/copy-paste/auto-augment/erasing。

结果：

| 指标 | V4 | Conservative Scout |
|---|---:|---:|
| false-alarm v1 FP/100 | 2 | 2 |
| public hard tiny recall | 0.7085 | 0.7735 |
| public hard fire recall | 0.8396 | 0.8615 |
| public hard smoke recall | 0.9252 | 0.9364 |
| public hard FP total | 271 | 590 |

解释：

- Conservative Scout 提升了召回，尤其是 tiny/fire/smoke recall。
- 但在 public hard holdout 上 FP 激增，说明模型变得更敏感。
- 因此它可以作为“recall-first”候选思路，但不能作为 balanced 主模型。

决策：

- 不把 scout 权重作为论文主模型。
- 正式 D1 必须使用全量 union 数据和更强的误报评估门槛。

## 4. 阈值扫描结论

V4 在 public hard holdout 上：

| conf | recall | tiny recall | FP/image | precision-like |
|---:|---:|---:|---:|---:|
| 0.10 | 0.8936 | 0.7691 | 0.884 | 0.7857 |
| 0.15 | 0.8836 | 0.7466 | 0.670 | 0.8271 |
| 0.20 | 0.8649 | 0.7085 | 0.542 | 0.8526 |
| 0.25 | 0.8450 | 0.6547 | 0.428 | 0.8774 |
| 0.30 | 0.8301 | 0.6211 | 0.344 | 0.8974 |
| 0.40 | 0.7888 | 0.5359 | 0.250 | 0.9196 |
| 0.50 | 0.7353 | 0.4148 | 0.174 | 0.9387 |

V4 在 false-alarm v1 上：

| conf | FP/100 |
|---:|---:|
| 0.10 | 2 |
| 0.15 | 2 |
| 0.20 | 2 |
| 0.25 | 1 |
| 0.30 | 1 |
| 0.40 | 1 |
| 0.50 | 1 |

部署建议：

- Recall-first：`conf=0.15-0.20`，适合早期预警和离线筛查。
- Balanced：`conf=0.25`，适合演示系统默认值。
- Conservative alarm：`conf=0.30-0.40`，适合减少误报，但会损失 tiny fire recall。

已执行：

- FastAPI 默认 `smoke_conf` 从 0.10 调整为 0.25。
- 后端服务已重启，`/health` 返回默认 `smoke_conf=0.25`。

## 5. 下一轮正式训练策略

不建议继续小 scout 微调。正式 D1 应采用：

- 数据：`datasets/stage6_mixed_3cls_union/data.yaml`
- 初始化：V4 best
- 学习率：`lr0=0.0001`
- 冻结：`freeze=10`
- 增强：关闭 `auto_augment` 和 `erasing`，mosaic 低或关闭
- 选择模型：不只看 mAP，必须同时看 fixed false-alarm、public hard tiny recall、FP/image

建议命令：

```powershell
python tools/train_yolo_api.py `
  --model D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/models/yolov8m-p2-fire-smoke-3cls.yaml `
  --data D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage6_mixed_3cls_union/data.yaml `
  --project D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/runs_stage6 `
  --name stage6_d1_union_conservative_e20 `
  --pretrained D:/Researching/Yolo/Yolo/models_stage4/v4_best.pt `
  --epochs 20 `
  --batch 4 `
  --imgsz 960 `
  --device 0 `
  --workers 4 `
  --patience 6 `
  --mosaic 0.05 `
  --mixup 0.0 `
  --copy-paste 0.0 `
  --close-mosaic 5 `
  --auto-augment none `
  --erasing 0.0 `
  --lr0 0.0001 `
  --lrf 0.2 `
  --weight-decay 0.0005 `
  --freeze 10 `
  --amp true `
  --save-period 2
```

通过门槛：

- false-alarm v1 FP/100 不高于 V4 的 2。
- public hard tiny recall 不低于 V4 的 0.7085，目标超过 0.75。
- public hard FP/image 不高于 V4 的 0.542，或在 recall 明显提升时给出 recall-first 模型定位。
- 若 FP 激增，则该模型只能作为 recall-first，不作为 balanced 主模型。

## 6. 论文支撑点更新

当前新增可写入论文的方法学支撑：

- 引入 verified-empty false-positive mining，避免人工全量复核。
- 使用三模型共识筛选高置信 hard negatives。
- 将 fixed false-alarm set 纳入模型选择，而不是只看 mAP。
- 明确给出 recall-first 与 balanced 两个部署工作点。

当前不能写成正向贡献的内容：

- 不能宣称 scout 微调降低了误报。
- 不能把小样本 hard-negative 微调作为最终方法。
- 不能只用 public hard recall 提升掩盖 FP 激增。

