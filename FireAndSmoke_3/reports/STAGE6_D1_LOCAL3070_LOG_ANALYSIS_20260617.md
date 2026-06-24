# Stage6 D1 本机训练日志分析

日期：2026-06-17

## 1. 当前结论

Stage6 D1 最初启动的本机训练没有完成。该进程在完成 2 个 epoch 后被停止，原因不是显存不足，而是训练实际学习率偏离了原计划。

已停止的运行目录：

- `runs_stage6_local3070/stage6_d1_local3070_conservative_e20_b2_i960`

保留原因：

- 该 run 可以作为一次失败先导实验，用来说明本机训练环境、数据加载和显存设置可行。
- 但它不适合作为论文主结果，也不应继续训练。

## 2. 停止原因

原计划使用保守微调：

- `lr0=0.0001`
- `freeze=10`
- `mosaic=0.05`
- `batch=2`
- `imgsz=960`

但是 `results.csv` 显示实际学习率明显偏高：

| epoch | mAP50 | mAP50-95 | lr/pg0 | lr/pg1 |
|---:|---:|---:|---:|---:|
| 1 | 0.83263 | 0.56601 | 0.00999974 | 0.00333325 |
| 2 | 0.83078 | 0.56745 | 0.01919970 | 0.00639992 |

这与 `lr0=0.0001` 不一致。日志中 `args.yaml` 和 trainer 参数显示 `optimizer=auto`，并且存在默认 `warmup_bias_lr=0.1`。判断为 Ultralytics 自动优化器与 warmup 设置覆盖了我们想要的低学习率微调策略。

与 V4 基线相比，前 2 个 epoch 的 mAP50-95 只有 0.56601 和 0.56745，低于 V4 的 0.58371。因此继续运行没有意义。

## 3. 已执行的修正

已修改训练脚本：

- `tools/train_yolo_api.py`

新增参数：

- `--optimizer`
- `--warmup-bias-lr`
- `--momentum`

这样可以显式关闭 `optimizer=auto` 的不确定性。

已启动 corrected D1：

- `runs_stage6_local3070/stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4`

关键设置：

- `optimizer=AdamW`
- `lr0=0.0001`
- `lrf=0.2`
- `warmup_epochs=0.0`
- `warmup_bias_lr=0.0001`
- `freeze=10`
- `batch=2`
- `imgsz=960`
- `amp=true`

日志确认 corrected run 的实际参数已经生效。

## 4. corrected run 当前状态

进程：

- PID：`26572`

当前状态：

- 已进入 epoch 1/20。
- 未出现 CUDA OOM。
- GPU 显存占用约 4.4GB。
- 训练速度约 7 到 8 it/s。

该 run 才是当前有效的本机 Stage6 D1。

## 4.1 epoch 1 结果更新

截至 2026-06-17 17:03，corrected run 已完成第 1 个 epoch，并开始第 2 个 epoch。

第 1 个 epoch 的验证结果如下：

| epoch | Precision | Recall | mAP50 | mAP50-95 | lr/pg0 | lr/pg1 | lr/pg2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.81682 | 0.78657 | 0.83402 | 0.56437 | 0.0001 | 0.0001 | 0.0001 |

判断：

- 低学习率已生效，之前 `optimizer=auto` 导致的学习率跑偏问题已经修复。
- 第 1 个 epoch 的 mAP50-95 为 0.56437，低于 V4 val mAP50-95 的 0.58371，差值约为 -0.01934。
- 第 1 个 epoch 结果不能作为最终结论，因为这是 hard-negative union 数据上的保守微调初期结果；但它提示本机 D1 不应无条件跑满 20 epoch。

当前决策：

- 继续训练到至少 epoch 3。
- epoch 3 后检查 mAP50-95、Precision、Recall 和学习率。
- 如果 epoch 3 的 mAP50-95 仍低于 0.57，且没有明显 recall/precision 收益，则不建议本机继续跑满 20 epoch，应停止并转服务器正式训练。
- 如果 epoch 3 接近或超过 0.58，再继续观察到 epoch 5，并在 `save_period=2` 生成 checkpoint 后进行固定集评估。

## 5. 训练完成后的自动评估

自动评估脚本已改为指向 corrected run：

- `tools/stage6_eval_d1_local3070.ps1`

训练结束并生成 `best.pt` 后，该脚本会输出：

- `reports/stage6_threshold_scan_d1_local3070_corrected_public_hard.csv`
- `reports/stage6_threshold_scan_d1_local3070_corrected_false_alarm_v2.csv`

评估门槛：

| 指标 | 门槛 |
|---|---|
| public hard tiny recall | 不低于 V4 的 0.70852，目标不低于 0.75 |
| public hard FP/image | 不高于 V4 的 0.542 |
| false-alarm v2 FP/1000 | balanced 目标不高于 S5 的 5；可接受目标不高于 V4 的 9 |
| smoke recall | 不低于 V4 的 0.92523 |

## 6. 下一步任务

1. 继续监控 corrected D1，至少等 epoch 1 完成后检查实际 `lr/pg*` 是否保持低学习率。
2. corrected D1 完成后，立即运行 `tools/stage6_eval_d1_local3070.ps1`。
3. 如果 corrected D1 仍不能同时降低误报并保持 tiny recall，则不要继续本机长训，转向服务器 4090 双卡正式训练，并保留本机结果作为先导实验。
4. 下一批论文实验需要补充外部基线：YOLOv8s/m/l、YOLOv10 或 YOLOv11、RT-DETR，以及切片推理实验。
