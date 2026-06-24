# Stage6 corrected D1 本机训练状态检查

检查时间：2026-06-17 17:05 左右  
运行目录：`runs_stage6_local3070/stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4`

## 1. 当前状态

corrected D1 训练尚未完成，当前进程仍在运行。

- 进程 PID：`26572`
- Python 路径：`D:\Python\python.exe`
- 当前阶段：已完成 epoch 1，正在训练 epoch 2/20
- 日志状态：未发现 CUDA OOM 或训练中断错误
- GPU 状态：RTX 3070 8GB 正在占用训练，温度约 64 摄氏度，GPU 利用率约 74%

## 2. corrected D1 参数是否生效

本次 corrected D1 的关键修正已经生效：

- `optimizer=AdamW`
- `lr0=0.0001`
- `lrf=0.2`
- `warmup_epochs=0.0`
- `warmup_bias_lr=0.0001`
- `freeze=10`
- `batch=2`
- `imgsz=960`
- `amp=true`

第 1 个 epoch 的 `results.csv` 显示三个参数组学习率均为 `0.0001`，说明前一次 `optimizer=auto` 和默认 warmup 导致学习率偏离的问题已经修复。

## 3. epoch 1 指标

| epoch | Precision | Recall | mAP50 | mAP50-95 | lr/pg0 | lr/pg1 | lr/pg2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.81682 | 0.78657 | 0.83402 | 0.56437 | 0.0001 | 0.0001 | 0.0001 |

与 Stage4 V4 验证集基线相比：

- V4 val mAP50：`0.84371`
- corrected D1 epoch 1 mAP50：`0.83402`
- 差值：`-0.00969`

- V4 val mAP50-95：`0.58371`
- corrected D1 epoch 1 mAP50-95：`0.56437`
- 差值：`-0.01934`

这说明 corrected D1 在第 1 个 epoch 还没有超过 V4 基线。但由于当前训练集加入了 hard-negative union 数据，并且 D1 是低学习率保守微调，第 1 个 epoch 不能直接判定失败。当前更合理的判断是继续观察到 epoch 3。

## 4. 论文实验角度的阶段判断

当前 corrected D1 的价值主要在于验证“加入无人机空场景 hard negative 后，是否能在不牺牲小目标火焰/烟雾召回的前提下降低误报”。因此不能只看常规验证集 mAP，还必须同时看：

- public hard holdout 的 tiny recall、fire recall、smoke recall、FP/image
- false-alarm v2 的 FP/1000 和 FP/image
- 与 V4、S5、No-P2 三个已有模型的固定阈值对比

如果 corrected D1 最终 mAP 略低，但 false-alarm v2 明显降低，同时 public hard 的 tiny recall 不下降，仍然可以作为论文中“鲁棒性增强训练”的有效实验支撑。反之，如果 mAP、召回和误报都没有优势，则不应把它作为主结果，只保留为失败先导实验。

## 5. 下一步执行决策

当前不建议立刻停止训练，也不建议现在运行 GPU 评估脚本，因为训练正在占用 RTX 3070。

下一步按以下门槛执行：

1. 继续训练到 epoch 3。
2. epoch 3 完成后重新读取 `results.csv`。
3. 如果 epoch 3 的 mAP50-95 仍低于 `0.57`，并且 Precision/Recall 没有明显改善，则停止本机长训练，保留该结果为本机先导实验。
4. 如果 epoch 3 的 mAP50-95 接近或超过 `0.58`，则继续训练到 epoch 5，并在生成 checkpoint 后进行固定评估。
5. 本机候选权重评估应使用：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\tools\stage6_eval_d1_local3070.ps1
```

评估输出应重点查看：

- `reports/stage6_threshold_scan_d1_local3070_corrected_public_hard.csv`
- `reports/stage6_threshold_scan_d1_local3070_corrected_false_alarm_v2.csv`

## 6. 当前结论

corrected D1 是当前有效的 Stage6 D1 本机训练，不是前一次学习率异常的无效运行。它已经修复学习率问题，但第 1 个 epoch 的指标仍低于 V4 基线，因此必须用 epoch 3 作为第一个硬性决策点。若 epoch 3 仍无改善，应把正式训练转移到 4090 服务器，并将本机结果作为参数修正和训练策略验证记录。

## 7. epoch 4 后的停止决策

2026-06-17 22:12 左右重新检查后，corrected D1 已完成 4 个 epoch。根据预设的 epoch 3 决策门槛，该运行没有达到继续本机长训的条件，因此已主动停止训练进程。

| epoch | Precision | Recall | mAP50 | mAP50-95 | lr/pg0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.81682 | 0.78657 | 0.83402 | 0.56437 | 0.000100 |
| 2 | 0.81802 | 0.78083 | 0.82674 | 0.56105 | 0.000096 |
| 3 | 0.82591 | 0.78052 | 0.83061 | 0.56324 | 0.000092 |
| 4 | 0.82499 | 0.77592 | 0.82626 | 0.55967 | 0.000088 |

停止原因：

- epoch 3 的 mAP50-95 为 `0.56324`，低于继续观察门槛 `0.57`。
- epoch 4 的 mAP50-95 进一步下降到 `0.55967`。
- Recall 从 epoch 1 的 `0.78657` 降到 epoch 4 的 `0.77592`。
- 继续在本机 3070 上跑满 20 epoch 的边际价值不足。

保留权重：

- `best.pt`：epoch 1 最佳验证指标权重
- `epoch2.pt`：中间 checkpoint
- `last.pt`：停止前最新权重

## 8. corrected D1 固定评估结果

训练停止后，已使用 `best.pt` 运行固定评估脚本：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\tools\stage6_eval_d1_local3070.ps1
```

输出文件：

- `reports/stage6_threshold_scan_d1_local3070_corrected_public_hard.csv`
- `reports/stage6_threshold_scan_d1_local3070_corrected_false_alarm_v2.csv`

### 8.1 public hard holdout

| conf | Recall | Tiny Recall | Fire Recall | Smoke Recall | FP/image | Precision-like |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.89079 | 0.76906 | 0.87011 | 0.94019 | 0.886 | 0.78474 |
| 0.15 | 0.87534 | 0.73991 | 0.85368 | 0.92710 | 0.676 | 0.82442 |
| 0.20 | 0.86266 | 0.70852 | 0.83568 | 0.92710 | 0.522 | 0.85699 |
| 0.25 | 0.84501 | 0.66592 | 0.81142 | 0.92523 | 0.420 | 0.87945 |
| 0.30 | 0.82239 | 0.63004 | 0.78247 | 0.91776 | 0.338 | 0.89819 |
| 0.40 | 0.78103 | 0.52242 | 0.72770 | 0.90841 | 0.254 | 0.91769 |
| 0.50 | 0.72918 | 0.40359 | 0.66119 | 0.89159 | 0.168 | 0.94026 |

与 V4 在 `conf=0.20` 的已知结果相比：

- tiny recall：D1 为 `0.70852`，与 V4 的 `0.70852` 持平。
- FP/image：D1 为 `0.522`，低于 V4 的 `0.542`。
- smoke recall：D1 为 `0.92710`，略高于 V4 的 `0.92523`。
- fire recall：D1 为 `0.83568`，略低于 V4 的 `0.83959`。

因此，D1 在 public hard 上没有提升小目标召回，但在保持 tiny recall 不下降的同时略微降低误报。

### 8.2 UAV false-alarm v2

| conf | False Positives | FP/image | FP Fire | FP Smoke |
|---:|---:|---:|---:|---:|
| 0.10 | 8 | 0.008 | 0 | 8 |
| 0.15 | 6 | 0.006 | 0 | 6 |
| 0.20 | 5 | 0.005 | 0 | 5 |
| 0.25 | 4 | 0.004 | 0 | 4 |
| 0.30 | 2 | 0.002 | 0 | 2 |
| 0.40 | 1 | 0.001 | 0 | 1 |
| 0.50 | 1 | 0.001 | 0 | 1 |

与已有模型在 `conf=0.20` 的已知结果相比：

- V4：`9/1000`
- S5：`5/1000`
- No-P2：`15/1000`
- corrected D1：`5/1000`

这说明 corrected D1 的 hard-negative 训练确实把空场景误报压到了 S5 的水平，并明显优于 V4 和 No-P2。

## 9. 论文实验定位

corrected D1 不适合作为当前主模型替代 V4/S5，因为常规验证 mAP50-95 低于 V4，且训练到 epoch 4 后没有上升趋势。

但它适合作为论文中的鲁棒性增强消融实验：

- 证明加入无人机空场景 hard negative 后，可以降低空场景误报。
- 证明 P2 小目标结构在 tiny recall 上仍然必要，因为 D1 在 `conf=0.20` 保持了 V4 的 tiny recall。
- 证明仅靠继续微调不能自动带来 mAP 提升，下一阶段需要更系统的数据配比、解冻策略和服务器规模训练。

## 10. 下一步执行方案

不继续本机 corrected D1 长训练。下一步应执行服务器正式训练 D2，目标是在保持 D1 误报收益的同时恢复或提升 mAP。

建议 D2 设置：

- 平台：4090 服务器
- 初始权重：V4 或 S5 最优权重
- 数据：继续使用 `stage6_mixed_3cls_union/data.yaml`
- 图像尺寸：`960` 或服务器可承受时尝试 `1280`
- batch：优先使用服务器显存做大 batch
- 冻结策略：先冻结 backbone 2 到 3 个 epoch，再全量解冻
- 学习率：继续使用显式 `AdamW`，避免 `optimizer=auto`
- hard negative：保留 verified-empty，后续再增加更高置信的误报样本
- 评估：每个候选权重必须跑 public hard、false-alarm v2 和原始 val 三套固定评估

D2 的论文验收门槛：

- false-alarm v2 在 `conf=0.20` 不高于 `5/1000`
- public hard tiny recall 不低于 `0.70852`
- public hard FP/image 不高于 `0.522`
- val mAP50-95 应恢复到接近或超过 V4 的 `0.58371`
- 如果 mAP 无法恢复，D2 只能作为鲁棒性消融，不能作为主结果

## 11. 已准备的服务器执行脚本

已新增服务器正式训练脚本：

- `tools/stage6_train_d2_server4090.sh`

该脚本采用两阶段训练：

1. phase1：从 S5 优先、V4 兜底的权重出发，冻结 backbone 训练 3 个 epoch，使模型先适配 hard-negative union 数据。
2. phase2：加载 phase1 的 `best.pt`，全量解冻并使用更小学习率继续训练 12 个 epoch，目标是保留误报收益并恢复 mAP。

默认参数：

- `DEVICE=0,1`
- `BATCH=24`
- `IMGSZ=960`
- `WORKERS=12`
- `optimizer=AdamW`
- `auto_augment=none`
- `mixup=0.0`
- `copy_paste=0.0`

在服务器项目目录下执行：

```bash
bash tools/stage6_train_d2_server4090.sh
```

如显存不足，可降低 batch：

```bash
BATCH=16 bash tools/stage6_train_d2_server4090.sh
```

如需要指定权重：

```bash
PRETRAINED=/home/uav/gu/path/to/s5_best.pt bash tools/stage6_train_d2_server4090.sh
```

已新增服务器通用评估脚本：

- `tools/stage6_eval_model_server.sh`

训练后执行：

```bash
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d2_server4090_phase2_unfreeze_s5_union_i960/weights/best.pt
```

该脚本会输出 public hard 和 false-alarm v2 两套阈值扫描结果，作为 D2 是否进入论文主表的依据。
