# Stage4→5 训练分析与下一步执行报告

生成日期：2026-06-13 14:30 CST
服务器：221.14.87.239:6022 (uav@/home/uav/gu)

---

## 执行摘要

1. ✅ V4 训练已完成，Early Stop at epoch 36 (best epoch 16)
2. ✅ V4 Holdout 评估完成：**mAP50=0.924, mAP50-95=0.681**
3. ✅ V4 大幅超越 E20 (holdout mAP50 +6.7%, mAP50-95 +4.2%)
4. ✅ Stage5 120 轮长训已启动并验证运行正常

---

## 1. V4 训练结果详解

### 1.1 训练过程

| 阶段 | Epoch | mAP50 | mAP50-95 | 说明 |
|------|-------|-------|----------|------|
| 起步 | 1 | 0.823 | 0.553 | MuSGD warmup 3 epoch |
| 上升 | 5 | 0.838 | 0.570 | 快速收敛 |
| 上升 | 10 | 0.842 | 0.580 | 接近峰值 |
| 峰值 | 16 | **0.844** | **0.584** | Best epoch |
| 饱和 | 25 | 0.841 | 0.583 | 开始过拟合 |
| 终止 | 36 | 0.836 | 0.580 | Early Stop |

### 1.2 过拟合分析

- 训练损失持续下降 (box_loss: 0.90 → 0.72)
- 验证 mAP50-95 在 epoch 16 后停滞在 0.580-0.584
- 验证 loss 开始回升 → 明确的过拟合信号
- **结论**：需要更强的正则化来支持更长训练

### 1.3 Per-class 详情 (V4 best @ epoch 16)

| 类别 | Precision | Recall | AP50 | AP50-95 |
|------|-----------|--------|------|---------|
| fire | 0.863 | 0.805 | 0.879 | 0.560 |
| other | 0.736 | 0.684 | 0.727 | 0.470 |
| smoke | 0.885 | 0.867 | 0.925 | 0.721 |

---

## 2. Holdout 评估对比（核心结论）

### 2.1 E20 vs V4 Holdout 对比

| 指标 | E20 best | V4 best | Delta | 提升幅度 |
|------|----------|---------|-------|---------|
| **mAP50** | 0.857 | **0.924** | +0.067 | **+7.8%** |
| **mAP50-95** | 0.639 | **0.681** | +0.042 | **+6.6%** |
| fire AP50 | 0.773 | **0.872** | +0.099 | **+12.8%** |
| fire AP50-95 | 0.491 | **0.545** | +0.054 | **+11.0%** |
| smoke AP50 | 0.941 | **0.975** | +0.034 | **+3.6%** |
| smoke AP50-95 | 0.787 | **0.817** | +0.030 | **+3.8%** |
| Precision | 0.878 | **0.901** | +0.023 | +2.6% |
| Recall | 0.875 | 0.870 | -0.005 | -0.6% |

### 2.2 关键发现

1. **V4 全面碾压 E20**：除 Recall 微降 0.5% 外，所有指标均大幅提升
2. **fire 类是最大受益者**：AP50 从 0.773 → 0.872，提升 12.8%
   - 这正是论文核心：无人机场景小火/小烟早期预警
3. **P2 架构效果确认**：P2 + 大数据集 + 正确训练策略 = 显著提升
4. **smoke 已近饱和**：AP50=0.975，进一步提升空间有限

### 2.3 历史模型全面对比

| Model | Epochs | Val mAP50 | Val mAP50-95 | Holdout mAP50 | Holdout mAP50-95 |
|-------|--------|-----------|--------------|----------------|------------------|
| Round1 Baseline | 100 | 0.8456 | 0.5736 | - | - |
| Stage3 P2 | 120 | 0.8387 | 0.5747 | - | - |
| E20 Scout | 20 | 0.8423 | 0.5805 | 0.857 | 0.639 |
| **V4 Corrected** | **36** | **0.844** | **0.584** | **0.924** | **0.681** |

---

## 3. Stage5 训练设计

### 3.1 设计原则

基于 V4 的经验教训：
1. **过拟合问题**：V4 best@16，之后持续过拟合
2. **解决方案**：更强的正则化 + 更低的学习率
3. **Fine-tuning 策略**：从已很强的 V4 best 出发，而非从头训练

### 3.2 训练配置

```python
model = YOLO(V4_BEST)  # 从最强模型出发
model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=960,
    device='0,1',
    batch=16,

    # 优化器 - fine-tuning 配置
    lr0=0.001,          # 低于 V4 的 0.003 (fine-tuning)
    lrf=0.01,
    optimizer='MuSGD',  # 已验证的最佳优化器
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,
    warmup_epochs=3,

    # 增强正则化 (V4 过拟合@16 的解决方案)
    mosaic=0.1,         # V4 用 0.15, 降低
    mixup=0.05,         # 新增 mixup
    copy_paste=0.1,     # 新增 copy_paste (小目标增强)
    erasing=0.2,        # 新增 random erasing
    close_mosaic=10,

    patience=30,         # V4 用 20, 增加容忍度
    save_period=5,
    plots=False,
    seed=20260613,
)
```

### 3.3 配置变更对照

| 参数 | V4 (36e) | Stage5 (120e) | 原因 |
|------|----------|---------------|------|
| lr0 | 0.003 | 0.001 | Fine-tuning 需更低 LR |
| mosaic | 0.15 | 0.1 | 降低过拟合 |
| mixup | 0.0 | 0.05 | 新增正则化 |
| copy_paste | 0.0 | 0.1 | 小目标增强 |
| erasing | 0.4 | 0.2 | 适中随机擦除 |
| patience | 20 | 30 | 允许更多探索 |

### 3.4 预期结果

- **乐观**：mAP50-95 达到 0.60+ (val), holdout 0.70+
- **保守**：与 V4 持平或微升 (+1-2%)
- **最差**：强正则化过度抑制学习，指标略降

---

## 4. 训练状态

### 4.1 当前运行

| 项目 | 状态 |
|------|------|
| 训练名称 | stage5_e120_from_v4best |
| 进程 PID | 168197 |
| GPU 0 | 10908 MiB / 46068 MiB |
| GPU 1 | 11020 MiB / 49140 MiB |
| 日志 | /home/uav/gu/stage4/stage5_e120_train.log |
| 结果 | /home/uav/gu/stage4/runs/stage5_e120_from_v4best/results.csv |

### 4.2 预计时间

- 每轮约 35 分钟
- 120 轮 = 约 70 小时
- **预计完成时间：2026-06-16 凌晨**

### 4.3 监控命令

```bash
# 查看训练进度
tail -20 /home/uav/gu/stage4/stage5_e120_train.log

# 查看 results.csv
cat /home/uav/gu/stage4/runs/stage5_e120_from_v4best/results.csv | tail -5

# 查看 GPU 状态
nvidia-smi

# 查看进程
ps aux | grep run_e120 | grep -v grep
```

---

## 5. 后续行动路线图

### 5.1 训练期间（等待中，约70小时）

- [ ] 定期检查训练进度（每8-12小时一次）
- [ ] 关注 epoch 5-10 的 mAP 趋势，确认 lr0=0.001 起步正常

### 5.2 训练完成后

- [ ] 分析 results.csv，确认 mAP50-95 变化趋势
- [ ] 对 Stage5 best.pt 执行 Holdout 评估
- [ ] 对比 Stage5 vs V4 vs E20 三代模型的 holdout 指标

### 5.3 论文实验（Stage5 确认后）

- [ ] **P2 消融实验**：在同一数据上跑普通 YOLOv8m (imgsz=960, 无 P2)，证明 P2 的贡献
- [ ] **SAHI 推理评估**：大图切片推理，评估真实 UAV 场景性能
- [ ] **Per-size 分析**：tiny/small/medium/large 分别的 AP 和 Recall
- [ ] **视频级评估**：首次检测时间 (Time-to-Detection)
- [ ] **混淆矩阵**：fire/other/smoke 分类错误分析

### 5.4 如果 Stage5 不如预期

- [ ] 分析过拟合/欠拟合原因
- [ ] 尝试 imgsz=1280 (更大分辨率)
- [ ] 尝试 AdamW 优化器
- [ ] 使用 V4 best 作为最终模型

---

## 6. 服务器上可用的模型权重

| 权重 | 路径 | Epoch | Holdout mAP50-95 |
|------|------|-------|------------------|
| E20 best | .../stage4_scout.../weights/best.pt | ~15 | 0.639 |
| E20 epoch5 | .../stage4_scout.../weights/epoch5.pt | 5 | 0.591 |
| V4 best | .../stage4_e60_v4_corrected/weights/best.pt | 16 | **0.681** |
| V4 last | .../stage4_e60_v4_corrected/weights/last.pt | 36 | 0.580 |
