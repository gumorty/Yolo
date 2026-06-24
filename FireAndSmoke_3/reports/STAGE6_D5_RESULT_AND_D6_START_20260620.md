# Stage6 D5 训练结果分析与 D6 高分辨率训练启动报告

日期：2026-06-20  
项目：无人机视角火焰/烟雾小目标检测  
服务器目录：`/home/uav/gu/projects/FireAndSmoke_3`

## 1. D5 训练完成状态

D5 已完成 8 个 epoch 训练，服务器 GPU 已在训练结束后恢复空闲。训练目录为：

```text
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960
```

关键文件：

```text
/home/uav/gu/stage6_d5_s5_hardneg_balanced_train.log
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960/results.csv
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960/weights/best.pt
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960/weights/last.pt
```

日志未发现 OOM、NaN、Traceback 或异常中断。训练过程中出现的 `0 corrupt` 属于 Ultralytics 数据扫描统计，不是异常。少量 duplicate labels 被自动移除，建议后续作为数据质量说明写入数据集清洗部分。

## 2. D5 主验证集结果

| 阶段 | best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| D2 | 10 | 0.83124 | 0.78457 | 0.82751 | 0.57055 |
| D3 | 6 | 0.82815 | 0.79070 | 0.82293 | 0.56743 |
| D4 | 11 | 0.82606 | 0.79046 | 0.81779 | 0.56384 |
| D5 | 6 | 0.83145 | 0.78723 | 0.83146 | 0.57379 |

D5 是目前 Stage6 服务器训练中主验证集 mAP50-95 最高的模型。相比 D3，D5 将 mAP50-95 从 0.56743 提升到 0.57379；相比 D4，提升更明显。这说明从 S5 best 重新做中等强度 hard negative 校准，比从 D3 继续全量解冻更适合保留主数据分布上的检测能力。

## 3. D5 固定评估结果

D5 已完成 public hard 与 false_alarm_v2 固定评估。核心结果如下：

| 模型 | conf | public hard recall | tiny recall | FP/image | false_alarm_v2 FP/1000 |
|---|---:|---:|---:|---:|---:|
| D3 | 0.20 | 0.87534 | 0.73991 | 0.534 | 5 |
| D4 | 0.20 | 0.88196 | 0.74888 | 0.584 | 4 |
| D5 | 0.20 | 0.87038 | 0.71973 | 0.520 | 5 |
| D5 | 0.25 | 0.85549 | 0.67937 | 0.420 | 4 |
| D5 | 0.30 | 0.83894 | 0.64574 | 0.348 | 2 |

D5 的优势是误报与主验证集 mAP 的平衡：在 conf=0.20 下，false_alarm_v2 达到 5 FP/1000，与 D3 持平，同时 public hard FP/image 降到 0.520，低于 D3 的 0.534 和 D4 的 0.584。D5 的主要不足是 tiny recall 下降到 0.71973，低于 D3/D4。

## 4. 当前论文支撑判断

D5 已经可以作为当前论文的强主模型候选，原因如下：

1. 它在 Stage6 中取得最高 mAP50-95，说明中等强度 hard negative 校准没有破坏主验证集性能。
2. 它在 false_alarm_v2 上达到 5 FP/1000，说明误报控制已经达到 D3 水平。
3. 它在 public hard 上 FP/image 最低，说明实际困难图像中的误报密度得到改善。

但如果论文要强调“无人机小目标早期火焰检测”，D5 的 tiny recall 仍不够理想。D3/D4 在 tiny recall 上更高，说明小目标召回仍然需要专项优化。因此，D5 适合作为综合平衡模型，下一步需要尝试提高小目标召回，并验证这种提升是否会牺牲 D5 的误报优势。

## 5. 下一轮训练决策：D6 高分辨率微调

根据 D5 的结果，下一轮不再继续增加 hard negative 强度，也不继续 D4 的全量解冻路线，而是从 D5 best 启动高分辨率微调：

- 起点权重：`D5 best.pt`
- 输入尺寸：1280
- batch：16
- epoch：6
- hard negative 重复强度：x4
- 学习率：0.00002
- 冻结：`freeze=10`
- 增强策略：关闭 mosaic、mixup、copy-paste、auto-augment、erasing
- 目标：提高 tiny recall 与定位质量，同时尽量保持 D5 的低误报和 mAP 优势

这轮训练的实验意义是验证“更高输入分辨率是否能改善无人机视角下的小目标火焰/烟雾检测”。如果 D6 的 tiny recall 高于 D5，同时 mAP50-95 和 FP/image 不显著恶化，则 D6 可成为最终主模型候选；如果 D6 提升 tiny recall 但误报或 mAP 恶化，则 D5 仍作为主模型，D6 作为分辨率消融实验。

## 6. D6 启动状态

D6 已在服务器后台启动：

```text
PID: 335517
日志: /home/uav/gu/stage6_d6_d5_highres_train.log
运行目录: /home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280
脚本: /home/uav/gu/projects/FireAndSmoke_3/tools/stage6_train_d6_d5_highres_server4090.sh
```

启动检查结果：

- DDP 初始化正常；
- AMP 检查通过；
- 训练/验证缓存读取正常；
- 已进入第 1/6 个 epoch；
- GPU 显存约 11.6GB/卡；
- 未出现 OOM 或异常中断。

查看实时日志：

```bash
tail -f /home/uav/gu/stage6_d6_d5_highres_train.log
```

查看 GPU 状态：

```bash
nvidia-smi
```

D6 完成后固定评估：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280/weights/best.pt
```

## 7. D6 完成后的判定标准

| 指标 | D5 基线 | D6 期望 |
|---|---:|---:|
| val mAP50-95 | 0.57379 | 不低于 0.570，越高越好 |
| public hard tiny recall@0.20 | 0.71973 | 明显高于 0.71973，接近或超过 D3 的 0.73991 |
| public hard FP/image@0.20 | 0.520 | 不显著高于 0.534 |
| false_alarm_v2 FP/1000@0.20 | 5 | 不高于 5 |

若 D6 同时满足以上条件，则论文主模型优先考虑 D6；若 D6 只提升 tiny recall 但牺牲误报或 mAP，则主模型仍选 D5，D6 作为高分辨率消融实验支撑。
