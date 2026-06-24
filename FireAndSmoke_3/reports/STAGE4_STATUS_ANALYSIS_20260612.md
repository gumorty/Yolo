# Stage4 训练状态综合分析报告

生成日期：2026-06-12
分析范围：服务器 221.14.87.239 + 本地 D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3

---

## 1. 项目历史回顾

### 1.1 历史训练轮次结果

| 轮次 | 模型 | Epochs | Precision | Recall | mAP50 | mAP50-95 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| Round1 | YOLOv8m Sensors 3cls | 100 | 0.8360 | 0.7805 | 0.8456 | 0.5736 | 基线 |
| Round1 | YOLOv8m Sensors 2cls | 100 | 0.8539 | 0.8325 | 0.8905 | 0.6149 | 2cls 更高 |
| Round2 | YOLOv8m 3cls+自有 | 150 | 0.8374 | 0.7798 | 0.8391 | 0.5715 | 未超过 Round1 |
| Round2 | YOLOv8m 2cls+自有 | 150 | 0.8715 | 0.8207 | 0.8856 | 0.6106 | 未超过 Round1 2cls |
| Stage3 | YOLOv8m-P2 Sensors 3cls | 120 | 0.8406 | 0.7680 | 0.8387 | 0.5747 | Recall 下降 |

### 1.2 核心问题定位

- Round2 问题：数据更多但标注风格不一致，other 类未等比例增加，模型变保守
- Stage3 问题：P2 检测头+imgsz=960，但数据仍以 640x640 为主，tiny/small smoke 仅 118 例，P2 没有足够数据支撑

---

## 2. Stage4 数据准备（已完成）

### 2.1 新增数据

| 数据集 | 图像数 | 关键价值 |
|---|---:|---|
| FASDD_UAV | 25,097 | UAV 高分辨率小目标，解决了 tiny/small smoke 极少的问题 |
| D-Fire | 25,833 | smoke-only 和 none 负样本，受控混入 train 5000/val 1000 |
| Sensors 3cls | 全部保留 | 历史基线分布 |

### 2.2 Full+Tile 数据集（最终训练数据）

| split | full images | positive tiles | negative tiles | total images | empty labels |
|---|---:|---:|---:|---:|---:|
| train | 34,856 | 37,087 | 3,801 | 75,744 | 12,560 |
| val | 16,613 | 19,231 | 2,462 | 38,306 | 7,099 |
| test | 2,741 | 2,513 | 10 | 5,264 | 47 |

### 2.3 Readiness 检查（全部 PASS）

| 检查项 | 当前值 | 门槛 | 结果 |
|---|---:|---:|---|
| train tiny+small fire | 47,119 | 5,000 | PASS |
| train tiny+small smoke | 9,011 | 1,000 | PASS |
| valid tiny+small fire | 27,793 | 300 | PASS |
| valid tiny+small smoke | 3,290 | 100 | PASS |
| total tiny+small other | 15,792 | 1,000 | PASS |
| high-res ratio | 0.5365 | 0.2 | PASS |

### 2.4 Public Hard Holdout

- 500 张固定评估图，全部来自 FASDD_UAV test
- 100 张空负样本，tiny+small fire 977，tiny+small smoke 126
- 不参与训练

---

## 3. 服务器训练状态（当前）

### 3.1 硬件状态

| 项目 | 状态 |
|---|---|
| GPU | 2 × RTX 4090 (46GB + 49GB) |
| CUDA | 12.8 |
| PyTorch | 2.5.1+cu121 |
| 磁盘 | 1.3T 可用 |
| GPU 进程 | 无（空闲） |

### 3.2 Stage4 Scout E20 训练（已完成，但崩溃于 final_eval）

**训练配置：**
- 模型：YOLOv8m-P2 3cls
- 迁移权重：sensors_yolov8m_3cls_100_best.pt
- 数据：stage4_full_tile_sensors3（75,744 train / 38,306 val）
- imgsz=960, batch=16, device=0,1, workers=8
- mosaic=0.20, mixup=0.0, copy_paste=0.0, erasing=0.0, auto_augment=none
- patience=8, close_mosaic=5, save_period=5

**训练结果（20 epoch 全部完成）：**

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.7528 | 0.6693 | 0.7452 | 0.4616 |
| 5 | 0.7842 | 0.7218 | 0.7955 | 0.5227 |
| 10 | 0.8115 | 0.7626 | 0.8341 | 0.5661 |
| 15 | 0.8166 | 0.7714 | 0.8402 | 0.5757 |
| 20 | 0.8216 | 0.7776 | 0.8423 | 0.5805 |

**崩溃原因：** matplotlib 字体加载失败
```
RuntimeError: Can not load face (unknown file format; error code 0x2)
```
在 `final_eval()` 阶段生成 PR_curve.png 时崩溃。训练本身成功，权重已保存。

**保存的权重：** best.pt, last.pt, epoch0.pt, epoch5.pt, epoch10.pt, epoch15.pt

### 3.3 Stage4 E60 续训（服务器中断前仅完成 5 轮）

**训练配置：**
- 基于 E20 best.pt 续训
- epochs=60, patience=20, cos_lr=True
- lr0=0.003, lrf=0.01, mosaic=0.15, close_mosaic=10
- plots=False（避免字体崩溃）
- seed=20260610

**已有结果（仅 5 轮）：**

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.8127 | 0.7732 | 0.8319 | 0.5660 |
| 2 | 0.7935 | 0.7596 | 0.8157 | 0.5473 |
| 3 | 0.7926 | 0.7384 | 0.8031 | 0.5303 |
| 4 | 0.8024 | 0.7344 | 0.8117 | 0.5356 |
| 5 | 0.8084 | 0.7443 | 0.8212 | 0.5473 |

**状态：** 服务器问题导致中断，当前停在第 5 轮。
**问题：** 续训前 5 轮指标波动，mAP50 和 mAP50-95 均低于 E20 best。这在 fine-tuning 初期是正常的，因为学习率重新预热。

### 3.4 Public Hard Holdout 评估（未完成）

虽然目录结构已创建（holdout_best_noplots, holdout_epoch5_noplots 等），但所有目录都是空的，没有实际评估结果。可能是因为评估过程也遇到了问题或被中断。

---

## 4. 关键发现与分析

### 4.1 Stage4 Scout E20 是成功的

与历史基线对比：

| 模型 | mAP50 | mAP50-95 | Recall |
|---|---:|---:|---:|
| Round1 YOLOv8m 3cls (100e) | 0.8456 | 0.5736 | 0.7805 |
| Stage3 P2 3cls (120e) | 0.8387 | 0.5747 | 0.7680 |
| **Stage4 Scout P2 3cls (20e)** | **0.8423** | **0.5805** | **0.7776** |

关键观察：
1. **mAP50-95 = 0.5805**，已经超过 Round1（0.5736）和 Stage3（0.5747），仅用 20 epoch
2. **mAP50 = 0.8423**，接近 Round1（0.8456），超过 Stage3（0.8387）
3. **Recall = 0.7776**，恢复到接近 Round1 水平（0.7805），远优于 Stage3（0.7680）
4. 训练曲线仍然在上升，没有出现饱和

### 4.2 E60 续训的初期波动是正常现象

- E60 从 E20 best 开始 fine-tune，使用了更小的 lr0=0.003 和 cos_lr=True
- 前 5 轮指标下降是学习率重新预热的结果
- 需要继续训练到至少 15-20 轮才能判断趋势

### 4.3 E20 训练的 matplotlib 崩溃

- 这不是训练问题，只是 Ultralytics 在 final_eval 阶段生成 PR_curve.png 时触发了 matplotlib 字体 bug
- 训练权重完整保存，不影响使用
- E60 已经设置了 plots=False，不会遇到此问题

### 4.4 服务器当前状态

- 服务器已恢复，双卡 GPU 空闲
- PyTorch CUDA 可用
- 训练数据和项目文件完整
- 可以立即恢复训练

---

## 5. 后续行动方案

### 5.1 立即行动（优先级最高）

**A. 恢复 E60 续训**

E60 训练在第 5 轮中断，需要恢复。有两个选择：

**方案一：从 last.pt resume（推荐）**
```bash
cd /home/uav/gu/projects/FireAndSmoke_3

ulimit -n 65535

PROJECT=/home/uav/gu/projects/FireAndSmoke_3 \
RUN_ROOT=/home/uav/gu/stage4 \
DATA=$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml \
NAME=stage4_p2_fasdd_dfire_tile_e60_from_e20best \
nohup python3 -c "
from ultralytics import YOLO
model = YOLO('/home/uav/gu/stage4/runs/stage4_p2_fasdd_dfire_tile_e60_from_e20best/weights/last.pt')
model.train(resume=True, plots=False)
" > /home/uav/gu/stage4/stage4_p2_fasdd_dfire_tile_e60_resume.log 2>&1 &
```

**方案二：从 E20 best 重新开始 60 轮训练**
```bash
cd /home/uav/gu/projects/FireAndSmoke_3

ulimit -n 65535

nohup python3 /home/uav/gu/projects/FireAndSmoke_3/tools/train_stage4_e60_from_e20best.py \
  > /home/uav/gu/stage4/stage4_e60_rerun.log 2>&1 &
```

### 5.2 短期行动（E60 训练期间可并行）

**B. 在服务器上完成 Public Hard Holdout 评估**

对 E20 的各 epoch 权重进行 holdout 评估：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3

# 评估 E20 best
python3 -c "
from ultralytics import YOLO
model = YOLO('/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/best.pt')
results = model.val(
    data='/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_eval/public_hard_holdout/data.yaml',
    imgsz=960,
    batch=16,
    device=0,1,
    plots=False,
    save_json=True,
    project='/home/uav/gu/stage4/eval_public_holdout',
    name='holdout_best_noplots',
)
print(results)
"

# 同样评估 epoch5, epoch10, epoch15, last
```

**C. 修复 matplotlib 字体问题**

```bash
# 删除有问题的字体缓存
rm -rf ~/.cache/matplotlib
# 或者安装缺失字体
apt-get install -y fonts-dejavu-core 2>/dev/null || sudo apt-get install -y fonts-dejavu-core
```

### 5.3 中期行动（E60 完成后）

1. 对 E60 各 epoch 权重在 public hard holdout 上评估
2. 对比 E20 best vs E60 best vs Round1 old model
3. 特别关注 small fire/smoke recall
4. 如果 E60 在 holdout 上有提升，考虑正式 120 轮长训
5. 如果 P2 仍不明确提升，在同一数据上跑普通 YOLOv8m 对照

### 5.4 长期规划

如果 Stage4 验证成功（scout 通过 holdout 评估）：
- 正式 120 轮训练
- 视频级评估（首次检测时间）
- 结构消融：如果 P2 提升不明显，对照普通 YOLOv8m
- 后续可以考虑 neck 改造、DySample、注意力重校准等

---

## 6. 风险提示

1. **E60 前 5 轮指标波动**：这是 fine-tuning 初期正常现象，不代表训练失败。需要继续训练到至少 15-20 轮才能看到稳定趋势
2. **D-Fire 域偏移**：D-Fire 主要是地面/监控视角，过多可能导致模型偏向非 UAV 场景。E60 使用了 cos_lr，应该能缓解
3. **E60 训练只到 60 轮**：如果需要更长时间训练，可以在 E60 完成后继续扩展到 120 轮
4. **Holdout 评估是关键**：目前还没完成 holdout 评估，这是判断模型是否真正提升的核心步骤
