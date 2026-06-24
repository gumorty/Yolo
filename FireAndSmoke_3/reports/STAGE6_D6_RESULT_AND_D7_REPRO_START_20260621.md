# Stage6 D6 训练结果分析与 D7 可重复性验证启动报告

日期：2026-06-21  
项目：无人机视角火焰/烟雾小目标检测  
服务器目录：`/home/uav/gu/projects/FireAndSmoke_3`

## 1. D6 完成状态

D6 已完成训练，服务器 GPU 已恢复空闲。训练目录为：

```text
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280
```

关键文件：

```text
/home/uav/gu/stage6_d6_d5_highres_train.log
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280/results.csv
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280/weights/best.pt
/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d6_d5_highres_x4_i1280/weights/last.pt
```

日志显示 D6 触发 EarlyStopping：原计划训练 6 个 epoch，实际完成 4 个 epoch。最佳结果出现在第 1 个 epoch，之后 3 个 epoch 没有改善，因此提前停止。训练过程没有发现 OOM、NaN、Traceback 或异常中断，`0 corrupt` 为数据扫描统计，不是错误。

## 2. D6 主验证集结果

| 阶段 | 输入尺寸 | best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| D3 | 960 | 5 | 0.82815 | 0.79070 | 0.82293 | 0.56743 |
| D4 | 960 | 10 | 0.82606 | 0.79046 | 0.81779 | 0.56384 |
| D5 | 960 | 5 | 0.83145 | 0.78723 | 0.83146 | 0.57379 |
| D6 | 1280 | 1 | 0.78868 | 0.77455 | 0.81097 | 0.48266 |

D6 的 mAP50-95 从 D5 的 0.57379 降到 0.48266，下降幅度约 0.09113。这个变化不是正常微调波动，而是明显的分布适配失败。结合训练日志，D6 在第 1 个 epoch 已经达到最好结果，后续继续下降，说明高分辨率 1280 微调没有改善当前数据与模型组合，反而削弱了定位质量和整体检测稳定性。

## 3. D6 固定评估结果

D6 已完成 public hard 与 false_alarm_v2 固定评估：

| 模型 | conf | public hard recall | tiny recall | FP/image | false_alarm_v2 FP/1000 |
|---|---:|---:|---:|---:|---:|
| D5 | 0.20 | 0.87038 | 0.71973 | 0.520 | 5 |
| D6 | 0.20 | 0.87148 | 0.71525 | 0.530 | 6 |
| D6 | 0.25 | 0.85880 | 0.68834 | 0.438 | 3 |
| D6 | 0.30 | 0.84832 | 0.65695 | 0.376 | 2 |

D6 在 conf=0.20 下 public hard recall 只比 D5 高 0.00110，但 tiny recall 从 0.71973 降到 0.71525，FP/image 从 0.520 升到 0.530，false_alarm_v2 从 5 FP/1000 升到 6 FP/1000。也就是说，D6 没有达成“提高小目标召回并保持低误报”的目标。

## 4. D6 对论文的意义

D6 不适合作为主模型候选，但它对论文仍有价值：

1. D6 可以作为输入分辨率/高分辨率微调的负向消融，证明简单提高输入尺寸并不能稳定提升无人机小目标火焰烟雾检测。
2. D6 说明当前数据中已经大量使用切片与 960 输入，继续提高到 1280 可能破坏训练分布与验证分布之间的平衡。
3. D6 支持一个更稳健的论文结论：本项目的提升主要来自 P2 小目标检测头、切片数据策略与 hard negative 校准，而不是单纯依赖更大输入分辨率。

因此，当前主模型仍应选择 D5，而不是 D6。

## 5. 下一步决策：D7 可重复性验证

按照顶会/SCI 论文实验标准，单次最优结果不足以支撑最终结论。D5 已经是当前综合最优模型，因此下一步不应继续盲目改结构或改分辨率，而应验证 D5 策略的稳定性。

D7 采用与 D5 相同的训练配置，只改变随机种子：

- 起点权重：`models_stage4/s5_best.pt`
- 数据：`stage4_full_tile_sensors3` + 两轮 verified empty hard negative
- hard negative 重复强度：x4
- 输入尺寸：960
- batch：24
- epoch：8
- 优化器：AdamW
- 学习率：0.00003
- 冻结：`freeze=10`
- seed：1
- deterministic：true

为支持该实验，已给训练封装 `tools/train_yolo_api.py` 增加 `--seed` 和 `--deterministic` 参数透传。

## 6. D7 启动状态

D7 已在服务器后台启动：

```text
PID: 401163
日志: /home/uav/gu/stage6_d7_s5_hardneg_balanced_seed1_train.log
运行目录: /home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d7_s5_hardneg_balanced_x4_i960_seed1
脚本: /home/uav/gu/projects/FireAndSmoke_3/tools/stage6_train_d7_s5_hardneg_balanced_seed_server4090.sh
```

启动检查结果：

- DDP 初始化正常；
- AMP 检查通过；
- `--seed 1 --deterministic true` 已进入训练进程命令；
- 已进入第 1/8 个 epoch；
- 双卡显存约 9.7GB/9.9GB；
- GPU 利用率约 86%/83%；
- 暂未发现 OOM 或异常中断。

查看实时日志：

```bash
tail -f /home/uav/gu/stage6_d7_s5_hardneg_balanced_seed1_train.log
```

D7 完成后固定评估：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d7_s5_hardneg_balanced_x4_i960_seed1/weights/best.pt
```

## 7. D7 完成后的判定标准

| 指标 | D5 seed0 | D7 seed1 期望 |
|---|---:|---:|
| val mAP50-95 | 0.57379 | 接近 D5，最好不低于 0.570 |
| public hard FP/image@0.20 | 0.520 | 不高于 0.534 |
| public hard tiny recall@0.20 | 0.71973 | 接近或高于 D5 |
| false_alarm_v2 FP/1000@0.20 | 5 | 不高于 5 |

若 D7 与 D5 接近，则可以将 D5/D7 作为同策略多 seed 证据，后续再补 seed2，形成均值与方差。如果 D7 明显偏离，则说明 D5 策略仍存在随机性敏感问题，论文中需要谨慎声明，并优先扩展评估集而不是继续堆训练轮次。
