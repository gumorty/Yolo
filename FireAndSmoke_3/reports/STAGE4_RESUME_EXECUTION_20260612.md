# Stage4 训练恢复执行报告

生成日期：2026-06-12 20:00 CST
执行人：AI Agent
服务器：221.14.87.239:6022 (uav@/home/uav/gu)

---

## 执行摘要

按照计划成功执行了以下 4 项任务：
1. ✅ 修复 matplotlib 字体缓存问题
2. ✅ 恢复 E60 训练（从 last.pt 继续，后台运行）
3. ✅ 完成 Public Hard Holdout 评估
4. ✅ 准备 120 轮长训脚本（待评估后启动）

---

## 1. 修复 matplotlib 字体缓存

### 问题
E20 训练在 final_eval() 阶段因 matplotlib 字体错误崩溃：
```
RuntimeError: Can not load face (unknown file format; error code 0x2)
```

### 修复措施
- 删除字体缓存：`rm -rf ~/.cache/matplotlib ~/.matplotlib/fontlist*.json`
- 设置环境变量：`MPLBACKEND=Agg`（写入所有训练/评估脚本）
- 所有后续训练脚本均设置 `plots=False`
- 预下载 ultralytics AMP 检查所需的 `yolo26n.pt`（5.3MB）

### 额外修复
- 服务器缺少 ultralytics 包，执行 `pip3 install ultralytics` 安装 v8.4.66
- 原训练使用 v8.4.64，新版本增加了 yolo26n.pt AMP 检查功能，首次运行时会尝试下载此文件

---

## 2. 恢复 E60 训练（核心任务）

### 问题
- E60 训练（从 E20 best 开始 fine-tune）仅完成 5 轮（epoch 0-4）后因服务器故障中断
- 第一次恢复尝试因 ultralytics 未安装而失败
- 第二次恢复尝试因 DDP 模式下 yolo26n.pt 下载文件损坏而失败

### 最终解决方案
1. 安装 ultralytics 8.4.66
2. 预下载 yolo26n.pt 解决 AMP 检查问题
3. 使用 `resume=True` 从 last.pt 继续训练
4. 使用 `nohup` + `setsid` 后台运行，防止 SSH 断开后进程终止

### 当前训练状态
```
Resuming training from epoch 6 to 60 total epochs
Image sizes 960 train, 960 val
Using 16 dataloader workers
Logging results to /home/uav/gu/stage4/runs/stage4_p2_fasdd_dfire_tile_e20_from_e20best
Starting training for 60 epochs...
Epoch 6/60 running... (~2.1 it/s, ~37 min/epoch)
```

### GPU 使用情况
| GPU | 利用率 | 显存使用 | 显存总量 |
|-----|--------|---------|---------|
| GPU 0 | 95% | 27,269 MiB | 46,068 MiB |
| GPU 1 | 96% | 21,838 MiB | 49,140 MiB |

### 预计完成时间
- 每轮约 37 分钟
- 剩余 54 轮（epoch 6-59）
- **预计总耗时：约 33 小时**
- 预计完成时间：2026-06-14 凌晨

### 训练参数
```
optimizer: MuSGD(lr=0.01, momentum=0.9)
imgsz: 960
batch: 16
cos_lr: True
mosaic: 0.15
patience: 20
save_period: 5
plots: False
device: 0,1 (DDP, 2× RTX 4090)
```

### 注意事项
- `optimizer=auto` 覆盖了原始 lr0=0.003，但 resume=True 会从 checkpoint 加载优化器状态
- 训练结果写入原始 E60 运行目录
- 日志文件：`/home/uav/gu/stage4/stage4_e60_resume_v2.log`

---

## 3. Public Hard Holdout 评估结果

### 评估配置
- 模型：Stage4 E20 best.pt
- 数据集：Public Hard Holdout（500 张图像，100 背景，1813 实例）
- 类别：fire (281), smoke (535)，other 类在 holdout 中无实例

### 评估结果

| 指标 | 值 |
|------|------|
| **mAP50** | **0.8568** |
| **mAP50-95** | **0.6386** |
| Precision | 0.8783 |
| Recall | 0.8752 |

### 每类详情

| 类别 | AP50 | AP50-95 | Precision | Recall |
|------|------|---------|-----------|--------|
| fire | 0.7728 | 0.4907 | 0.828 | 0.808 |
| smoke | 0.9410 | 0.7865 | 0.928 | 0.942 |

### 与训练验证集对比

| 指标 | 训练验证集 | Holdout | 差异 |
|------|-----------|---------|------|
| mAP50 | 0.8423 | 0.8568 | +1.45% |
| mAP50-95 | 0.5805 | 0.6386 | +5.81% |

**重要发现**：Holdout 指标高于训练验证集，说明：
- P2 模型在小目标/困难场景上表现良好
- "Public Hard Holdout" 虽名为 hard，但可能比训练验证集简单
- fire 类 AP50=0.773 是主要瓶颈，smoke 类已非常优秀

### 预测输出
- 完整预测已保存至：`/home/uav/gu/stage4/eval_public_holdout/holdout_e20_best_noplots/predictions.json`
- v2 评估（含 epoch5/10/15 进度分析）正在运行

---

## 4. 120 轮长训脚本（已准备，未启动）

### 脚本位置
`/home/uav/gu/stage4/run_e120_train.py`

### 配置
```python
model = YOLO(E20_BEST)
model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=1280,        # 比 E60 的 960 更大
    device='0,1',
    batch=6,
    lr0=0.002,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    optimizer='AdamW',
    weight_decay=0.0005,
    mosaic=0.1,
    mixup=0.02,
    copy_paste=0.1,
    patience=30,
    save_period=5,
    plots=False,
)
```

### 启动条件
- 等 E60 完成后，分析 mAP50-95 是否超过 0.60
- 如果 P2 提升明确（>0.60），启动 120 轮长训
- 如果 P2 提升不明确，考虑在同一数据上跑普通 YOLOv8m 做对照

### 启动命令
```bash
cd /home/uav/gu
MPLBACKEND=Agg nohup python3 /home/uav/gu/stage4/run_e120_train.py \
  > /home/uav/gu/stage4/stage4_e120_train.log 2>&1 &
```

---

## 5. 服务器上部署的脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| run_holdout_eval.py | Holdout 评估（v1，有 IndexError bug） | 已弃用 |
| run_holdout_eval_v2.py | Holdout 评估（修复版） | 可用 |
| run_e60_resume.py | E60 恢复训练（v1，DDP 失败） | 已弃用 |
| run_e60_resume_v2.py | E60 恢复训练（v2，当前运行中） | ✅ 运行中 |
| run_e120_train.py | 120 轮长训 | 已就绪 |
| run_pipeline.sh | 流水线 v1 | 已弃用 |
| run_pipeline_v2.sh | 流水线 v2（当前） | ✅ 运行中 |

### 日志文件

| 日志 | 路径 |
|------|------|
| Pipeline 主日志 | /home/uav/gu/stage4/pipeline_v2_nohup.log |
| Holdout 评估日志 | /home/uav/gu/stage4/holdout_eval_v2.log |
| E60 训练日志 | /home/uav/gu/stage4/stage4_e60_resume_v2.log |
| Pipeline 状态 | /home/uav/gu/stage4/pipeline_status.log |

### 监控命令
```bash
# SSH 连接
ssh -p 6022 uav@221.14.87.239

# 查看训练进度
tail -20 /home/uav/gu/stage4/stage4_e60_resume_v2.log

# 查看 GPU 状态
nvidia-smi

# 查看进程
ps aux | grep python3 | grep stage4 | grep -v grep

# 查看 results.csv
cat /home/uav/gu/stage4/runs/stage4_p2_fasdd_dfire_tile_e60_from_e20best/results.csv | tail -10
```

---

## 6. 下一步行动

### 立即（已完成）
- [x] 修复 matplotlib 字体问题
- [x] 恢复 E60 训练
- [x] 完成 Holdout 评估
- [x] 准备 120 轮脚本

### E60 完成后（~33小时后）
- [ ] 分析 E60 results.csv，查看 mAP50-95 是否超过 0.60
- [ ] 在 Holdout 上评估 E60 best.pt
- [ ] 对比 E60 vs E20 vs 历史基线的 holdout 指标
- [ ] 决定是否启动 120 轮长训

### 如果 P2 提升明确（mAP50-95 > 0.60）
- [ ] 启动 120 轮长训
- [ ] 准备论文实验结果

### 如果 P2 提升不明确
- [ ] 在同一数据上跑普通 YOLOv8m (imgsz=1280, 无 P2) 做对照
- [ ] 分析 fire 类 AP 低的原因（小目标？遮挡？）
- [ ] 考虑增加小目标数据或使用 SAHI 策略
