# Stage6 D4 诊断与 D5 下一轮训练启动报告

日期：2026-06-20  
项目：无人机视角火焰/烟雾小目标检测  
服务器目录：`/home/uav/gu/projects/FireAndSmoke_3`  
本地项目目录：`D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3`

## 1. 当前训练状态概览

服务器双 RTX 4090 在 D5 启动前处于空闲状态，显存占用约 1 MiB，GPU 利用率为 0%。已完成的 Stage6 服务器训练包括：

| 阶段 | 训练入口 | 主要策略 | 结果状态 |
|---|---|---|---|
| D2 | S5 权重继续训练 | Stage4 主数据 + 已验证空背景 hard negative，先冻结再解冻 | 已完成 |
| D3 | D2 best 继续训练 | hard negative 强校准，重复强度 x8，冻结骨干 | 已完成 |
| D4 | D3 best 继续训练 | 全量解冻，低学习率恢复主验证集 mAP | 已完成 |
| D5 | S5 best 重新校准 | hard negative 中等强度 x4，冻结骨干 | 已启动 |

## 2. D2/D3/D4 训练日志与收敛分析

### D2

D2 Phase2 共完成 12 个 epoch，最佳 mAP50-95 为 0.57055，最后一轮 mAP50-95 为 0.56979。最佳值与末轮值非常接近，说明 D2 后期已经进入平台区，继续同策略长训的边际收益较小。D2 在 public hard 上取得较高召回，但 false_alarm_v2 在 conf=0.20 下为 8 FP/1000，误报抑制未达到论文主模型的理想目标。

### D3

D3 共完成 8 个 epoch，最佳 mAP50-95 为 0.56743，最后一轮 mAP50-95 为 0.56734。D3 相比 D2 的主验证集 mAP 略降，但 public hard 与 false_alarm_v2 的平衡更好。conf=0.20 时，public hard recall 为 0.87534，tiny recall 为 0.73991，FP/image 为 0.534；false_alarm_v2 为 5 FP/1000。该结果说明强 hard negative 校准确实降低了误报，并提升了困难集小目标召回，但代价是常规验证集 mAP50-95 没有恢复到 S5/V4 基线水平。

### D4

D4 共完成 10 个 epoch，最佳 mAP50-95 为 0.56384，低于 D2 和 D3。固定评估补测结果如下：

| 模型 | public hard conf=0.20 recall | tiny recall | FP/image | false_alarm_v2 FP/1000 |
|---|---:|---:|---:|---:|
| D2 | 0.86983 | 0.73543 | 0.548 | 8 |
| D3 | 0.87534 | 0.73991 | 0.534 | 5 |
| D4 | 0.88196 | 0.74888 | 0.584 | 4 |

D4 的 false_alarm_v2 降到 4 FP/1000，tiny recall 也提高到 0.74888，但 public hard FP/image 上升到 0.584，且主验证集 mAP50-95 进一步下降到 0.56384。这说明 D4 的全量解冻策略偏向提高召回和压低纯空背景误报，但没有改善整体定位质量与精度平衡，不适合作为当前论文主模型候选。

## 3. 异常记录

当前检查未发现 OOM、NaN、Traceback 或训练中断。日志中出现的 `0 corrupt` 是 Ultralytics 数据扫描的正常统计，不代表数据损坏。D5 启动后的 AMP 检查通过，训练/验证缓存扫描完成，数据加载和优化器初始化正常。

需要注意的是，训练日志显示部分图像存在重复标签并被自动移除。这不是阻断性异常，但在论文级实验中建议后续加入数据质量统计表，把重复标签、空背景比例、类别分布作为数据集清洗证据。

## 4. 是否支撑论文论点

目前实验已经能支撑以下论文论点：

1. P2/高分辨率/切片策略对无人机小目标火焰烟雾检测有效，V4、S5、No-P2 与 Stage6 的对比可以构成结构与数据策略消融。
2. hard negative 校准能显著改善误报控制。D3 相比 D2 将 false_alarm_v2 从 8 FP/1000 降到 5 FP/1000，同时 public hard tiny recall 达到 0.73991。
3. 召回、误报、mAP 之间存在明显权衡。D4 提高了 tiny recall 并降低纯空背景误报，但 public hard FP/image 和主验证集 mAP50-95 变差，这一结果可以作为论文中“鲁棒性校准不可只看单一指标”的证据。

当前还不足以作为 SCI 一区/二区论文的最终实验闭环，主要短板是：Stage6 的 mAP50-95 仍低于 S5/V4 基线；误报评估集规模还需要扩展到更多非火烟场景；最终主模型需要同时满足常规验证集 mAP、困难集 tiny recall 和低误报三个指标。

## 5. 下一轮训练决策：D5

基于 D2/D3/D4 的趋势，下一轮不再沿 D4 全量解冻路线继续训练，而是回到 S5 best 权重，采用中等强度 hard negative 校准：

- 起点权重：`/home/uav/gu/projects/FireAndSmoke_3/models_stage4/s5_best.pt`
- 数据：`stage4_full_tile_sensors3` + 两轮 verified empty hard negative
- hard negative 重复强度：x4
- 训练轮数：8 epoch
- 输入尺寸：960
- batch：24
- GPU：`0,1`
- 优化器：AdamW
- 初始学习率：0.00003
- 冻结：`freeze=10`
- 增强：关闭 mosaic、mixup、copy-paste、auto-augment、erasing

这个 D5 设计的目标是验证一个关键论文假设：中等强度 hard negative 校准是否能在保留 S5 主验证集 mAP 的同时，将 false_alarm_v2 压到 D3/D4 水平。若 D5 成功，它可以成为主模型候选；若 D5 不成功，则 D2/D3/D4/D5 共同构成完整的 hard negative 强度消融证据。

## 6. D5 启动状态

D5 已在服务器后台启动：

- 进程 PID：`310689`
- 训练日志：`/home/uav/gu/stage6_d5_s5_hardneg_balanced_train.log`
- 运行目录：`/home/uav/gu/projects/FireAndSmoke_3/runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960`
- 服务器脚本：`/home/uav/gu/projects/FireAndSmoke_3/tools/stage6_train_d5_s5_hardneg_balanced_server4090.sh`
- 本地脚本：`D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\tools\stage6_train_d5_s5_hardneg_balanced_server4090.sh`

启动后 GPU 占用约为 GPU0 10027 MiB、GPU1 9903 MiB，利用率约 59%/52%。日志显示已经进入第 1/8 个 epoch，初始 batch 的 box_loss、cls_loss、dfl_loss 正常波动，暂无 OOM 或异常中断。

## 7. 查看与后续评估指令

查看实时日志：

```bash
tail -f /home/uav/gu/stage6_d5_s5_hardneg_balanced_train.log
```

查看 GPU 状态：

```bash
nvidia-smi
```

D5 完成后执行固定评估：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
bash tools/stage6_eval_model_server.sh runs_stage6_server4090/stage6_d5_s5_hardneg_balanced_x4_i960/weights/best.pt
```

评估后重点比较：

| 指标 | 期望方向 | 判定目标 |
|---|---|---|
| val mAP50-95 | 越高越好 | 尽量接近 S5，至少高于 D3/D4 |
| public hard tiny recall | 越高越好 | 不低于 D3 的 0.73991 为佳 |
| public hard FP/image | 越低越好 | 不高于 D3 的 0.534 为佳 |
| false_alarm_v2 FP/1000 | 越低越好 | 不高于 5 为佳 |

## 8. 对论文实验章节的意义

D5 是当前实验链条中最关键的一轮平衡实验。它不是简单追加训练，而是在 D2/D3/D4 暴露出“误报降低与 mAP 回落”矛盾之后，用更合理的 hard negative 强度重新寻找主模型候选。若 D5 达到预期，论文可以将 S5 作为强基线、D3/D4 作为误报强化消融、D5 作为综合最优模型；若 D5 未达到预期，论文也能清楚呈现方法局限，并指导下一阶段改为数据侧清洗、场景扩充或分阶段阈值校准。
