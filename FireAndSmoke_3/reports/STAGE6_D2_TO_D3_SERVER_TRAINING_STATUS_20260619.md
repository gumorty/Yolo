# Stage6 D2 训练分析与 D3 启动记录

生成时间：2026-06-19  
服务器目录：`/home/uav/gu/projects/FireAndSmoke_3`  
服务器资源：双 RTX 4090  
当前目标：根据 D2 训练结果启动下一轮专项训练，使实验更接近论文主结果要求。

## 1. D2 训练状态

D2 两阶段训练已经完成。

### Phase 1

运行目录：

`runs_stage6_server4090/stage6_d2_server4090_phase1_freeze_s5_union_i960`

训练策略：

- 从 S5/V4 权重启动
- 冻结 backbone
- 训练 3 epoch
- `batch=24`
- `imgsz=960`
- `optimizer=AdamW`
- `lr0=0.0001`

Phase 1 最佳结果出现在 epoch 3：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 3 | 0.82579 | 0.78977 | 0.82976 | 0.56864 |

### Phase 2

运行目录：

`runs_stage6_server4090/stage6_d2_server4090_phase2_unfreeze_s5_union_i960`

训练策略：

- 从 Phase 1 `best.pt` 启动
- 全量解冻
- 训练 12 epoch
- `batch=24`
- `imgsz=960`
- `optimizer=AdamW`
- `lr0=0.00006`

Phase 2 最佳 mAP50-95 出现在 epoch 9：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 9 | 0.83124 | 0.78457 | 0.82751 | 0.57055 |

最终 epoch 12 结果：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 12 | 0.83054 | 0.78803 | 0.82570 | 0.56979 |

## 2. D2 固定评估结果

已使用 D2 Phase 2 `best.pt` 进行固定阈值评估。

输出文件位于服务器：

- `reports/stage6_threshold_scan_stage6_d2_server4090_phase2_unfreeze_s5_union_i960_public_hard.csv`
- `reports/stage6_threshold_scan_stage6_d2_server4090_phase2_unfreeze_s5_union_i960_false_alarm_v2.csv`

### 2.1 public hard holdout

`conf=0.20` 下结果：

| 模型 | Recall | Tiny Recall | Fire Recall | Smoke Recall | FP/image |
|---|---:|---:|---:|---:|---:|
| V4 | 0.86486 | 0.70852 | 0.83959 | 0.92523 | 0.542 |
| S5 | 0.87148 | 0.72197 | 0.84742 | 0.92897 | 0.538 |
| D1 corrected | 0.86266 | 0.70852 | 0.83568 | 0.92710 | 0.522 |
| D2 server | 0.86983 | 0.73543 | 0.84585 | 0.92710 | 0.548 |

D2 的优势是 tiny recall 达到 `0.73543`，高于 V4、S5 和 D1；问题是 FP/image 上升到 `0.548`，高于 D1 的 `0.522`，也略高于 V4/S5。

### 2.2 false-alarm v2

`conf=0.20` 下结果：

| 模型 | FP/1000 | FP/image |
|---|---:|---:|
| V4 | 9 | 0.009 |
| S5 | 5 | 0.005 |
| D1 corrected | 5 | 0.005 |
| D2 server | 8 | 0.008 |

D2 在 false-alarm v2 上优于 V4，但没有达到 D1/S5 的误报水平。它说明两阶段训练提高了小目标召回，但 hard-negative 抑制不足。

## 3. D2 对论文的支撑与不足

D2 可以支撑以下结论：

- 服务器两阶段训练能够恢复一部分主验证指标，mAP50-95 从本机 D1 的 `0.56437` 提高到 `0.57055`。
- D2 明显提升 public hard tiny recall，说明小目标召回方向有效。
- D2 没有达到最终主模型标准，因为 mAP50-95 仍低于 V4 的 `0.58371`，且 false-alarm v2 没有达到 `<=5/1000` 的验收门槛。

因此 D2 目前更适合写成“小目标召回增强候选”，不能直接作为最终主模型。

## 4. 下一轮任务选择

根据 D2 的表现，下一轮不优先做 `imgsz=1280` 高分辨率训练。原因是 D2 的主要短板不是 tiny recall，而是误报回升。

当前启动的下一轮为：

`D3 hard-negative calibration`

目标：

- 从 D2 best 继续训练。
- 加强 verified-empty hard negative 的采样权重。
- 将 false-alarm v2 从 D2 的 `8/1000` 压回 `<=5/1000`。
- 尽量保持 D2 的 public hard tiny recall `0.73543`。

## 5. D3 训练配置

服务器脚本：

`tools/stage6_train_d3_hardneg_server4090.sh`

服务器日志：

`/home/uav/gu/stage6_d3_hardneg_train.log`

运行目录：

`runs_stage6_server4090/stage6_d3_hardneg_calibration_x8_i960`

关键参数：

- 初始权重：`runs_stage6_server4090/stage6_d2_server4090_phase2_unfreeze_s5_union_i960/weights/best.pt`
- 训练轮数：`8`
- `batch=24`
- `imgsz=960`
- `device=0,1`
- `optimizer=AdamW`
- `lr0=0.00003`
- `lrf=0.2`
- `freeze=10`
- `mosaic=0.0`
- `mixup=0.0`
- `copy_paste=0.0`
- `HARD_REPEAT=8`

重要说明：D3 只重复采样 `verified_empty_fp_round01` 和 `verified_empty_fp_consensus_round02`，没有把 `false_alarm_v2` 固定评估集加入训练，因此不会污染论文测试集。

## 6. D3 当前启动状态

D3 已经在服务器后台启动，并进入实际 batch 训练。

启动命令：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
nohup bash tools/stage6_train_d3_hardneg_server4090.sh > /home/uav/gu/stage6_d3_hardneg_train.log 2>&1 &
```

当前观察到的状态：

- 进程存在：`python tools/train_yolo_api.py ... stage6_d3_hardneg_calibration_x8_i960`
- DDP 进程存在：`torch.distributed.run --nproc_per_node 2`
- 已进入 epoch `1/8`
- 训练 batch 进度已超过 `10%`
- 双卡 GPU 利用率约 `80%` 以上
- 未发现 OOM 或路径错误

## 7. 监控命令

查看训练日志：

```bash
tail -80 /home/uav/gu/stage6_d3_hardneg_train.log
```

查看 GPU：

```bash
nvidia-smi
```

查看训练指标：

```bash
cat /home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d3_hardneg_calibration_x8_i960/results.csv
```

查看进程：

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E "stage6_d3|train_yolo|torch.distributed.run" | grep -v grep
```

## 8. D3 完成后的评估标准

D3 完成后必须执行：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d3_hardneg_calibration_x8_i960/weights/best.pt
```

D3 是否继续作为候选模型取决于以下门槛：

| 指标 | 最低要求 | 理想目标 |
|---|---:|---:|
| false-alarm v2 FP/1000 | <= 5 | < 5 |
| public hard tiny recall | >= 0.72197 | 接近或超过 D2 的 0.73543 |
| public hard FP/image | <= 0.522 | 低于 D1 的 0.522 |
| val mAP50-95 | 不明显低于 D2 的 0.57055 | 接近 V4 的 0.58371 |

如果 D3 达到误报目标但 mAP 仍不足，它可作为“保守告警模型”进入论文；如果 D3 同时保持 tiny recall 和降低误报，它将是比 D2 更强的主候选。
