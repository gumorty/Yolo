# 论文实验补全方案：从当前状态到 SCI 可投稿标准

生成日期：2026-06-21
基于：Stage6 D2-D7 全部实验结果 + 顶会/顶会论文调研

---

## 一、当前实验资产盘点

### 1.1 已完成实验

| 实验 | 模型 | val mAP50-95 | public hard tiny recall | FP/1000 (v2) | 可重复性 | 论文角色 |
|------|------|:---:|:---:|:---:|:---:|------|
| V4 | YOLOv8m-P2 | **0.584** | 0.709 | 9 | — | 主基线 |
| S5 | V4 续训 | 0.584 | 0.722 | 5 | — | 微调对照 |
| No-P2 | YOLOv8m (3头) | 0.580 | 0.706 | 15 | — | 结构消融 |
| D1 | V4+HN(冻结) | 0.564 | 0.709 | 5 | — | 本机先导 |
| D2 | S5+HN(两阶段) | 0.571 | 0.735 | 8 | — | 服务器实验 |
| D3 | D2+HN×8(冻结) | 0.567 | 0.740 | 5 | — | 误报压制 |
| D4 | D3+解冻 | 0.564 | — | — | — | 失败实验 |
| **D5** | **S5+HN×4(冻结)** | **0.574** | **0.720** | **5** | ✅ | **主候选** |
| D6 | D5+imgsz1280 | 0.483 | 0.715 | 6 | — | 负向消融 |
| **D7** | **S5+HN×4(seed=1)** | **0.574** | **0.720** | **5** | ✅ | **可重复性** |

### 1.2 已有评估资产

| 评估集 | 图像数 | 内容 | 状态 |
|--------|:---:|------|:---:|
| 原始 val | 38,306 | 常规验证集 | ✅ |
| public hard holdout | 500 | 困难场景固定测试 | ✅ |
| uav_false_alarm_v2 | 1,000 | 纯空场景误报测试 | ✅ |
| 阈值扫描 | 7 个 conf | recall/FP 权衡曲线 | ✅ |

### 1.3 与顶会/顶会论文标准对比的 GAP

| 论文实验模块 | 顶刊要求 | 当前状态 | GAP |
|------------|---------|---------|-----|
| **外部基线对比** | YOLOv8n/s/m/l + v10/v11 + RT-DETR | ❌ 仅内部模型 | **严重缺失** |
| **消融实验** | P2+HN+分辨率+推理策略 | ⚠️ P2+HN+分辨率已补，缺 SAHI | 中等 |
| **Per-size AP** | AP_small/medium/large | ❌ 完全缺失 | **严重缺失** |
| **效率指标** | FPS+Params+FLOPs | ❌ 完全缺失 | **严重缺失** |
| **视频级评估** | FP/min+首次检测+时序稳定 | ❌ 完全缺失 | **严重缺失** |
| **统计可靠性** | 3 seed 均值+方差 | ⚠️ 2 seed 已完成 | 小 |
| **失败案例分析** | FP/FN 可视化+混淆矩阵 | ❌ 缺失 | 中等 |
| **误报评估协议** | 固定空场景 FP/image | ✅ 已有 1000 图 | 已达标 |
| **部署阈值策略** | recall-first/balanced/conservative | ✅ 已有 7 阈值扫描 | 已达标 |
| **数据集说明** | 来源+划分+质量控制 | ✅ 已有 | 已达标 |

---

## 二、顶会/顶会论文实验标准（基于调研）

### 2.1 FireRescue (arXiv 2025) 实验标准

- **对比模型**：19 个（YOLOv3→v12 + RT-DETR + Mamba-YOLO）
- **消融**：6 组（逐步叠加模块）
- **指标**：mAP50/mAP50-95/Precision/Recall/Params/FLOPs
- **数据集**：15,980 图，8:1:1 划分，专家标注
- **可视化**：Grad-CAM 热力图

### 2.2 Nature Sci Rep 2025 (Multiscale wildfire) 实验标准

- **对比模型**：4 个跨架构模型
- **消融**：7 组（3 模块全组合）
- **指标**：Precision/Recall/FPS/FLOPs/Params
- **数据集**：21,963 图（D-Fire+自采），8:1:1
- **FPS**：64.5 FPS（含部署效率报告）

### 2.3 FG-YOLO (Springer 2025) 实验标准

- **指标**：Accuracy/Recall/mAP50/mAP50-95/FPS/Params
- **FPS**：76.51 FPS
- **参数量**：2.8M（轻量化报告）
- **小目标**：明确报告小目标 AP 改善

### 2.4 共性要求总结

```
必选项（所有论文都有）：
  ✅ mAP50 + mAP50-95 + Precision + Recall
  ✅ FPS + Params + FLOPs
  ✅ 多版本 YOLO 基线对比（至少 4-5 个外部模型）
  ✅ 消融实验（逐步叠加）
  ✅ 数据集划分说明

加分项（多数论文有）：
  ✅ 小目标 AP 分解
  ✅ 可视化检测案例（成功+失败）
  ✅ Grad-CAM 注意力可视化

差异化（我们独有）：
  ✅ 误报评估协议（false_alarm_v2, 1000 图）
  ✅ 部署阈值策略（recall-first/balanced/conservative）
  ✅ Hard negative mining 方法
  ✅ 可重复性验证（多 seed）
```

---

## 三、实验补全计划（按优先级排序）

### P0：必须完成（论文无法投稿的硬伤）

#### 3.1 外部基线训练（预计 2-3 天服务器时间）

**目标**：在统一数据集上训练至少 5 个外部基线模型。

| 模型 | 参数量 | 训练时间 | 来源 | 论文作用 |
|------|:---:|:---:|------|------|
| YOLOv8n | 3.2M | ~4h | ultralytics 官方 | 轻量化对照 |
| YOLOv8s | 11.2M | ~6h | ultralytics 官方 | 中等规模对照 |
| YOLOv8m (标准, 无 P2) | 25.9M | ~8h | ultralytics 官方 | 同规模无改进对照 |
| YOLOv10m | 24.6M | ~8h | ultralytics 官方 | 跨版本对照 |
| RT-DETR-l | 32M | ~12h | ultralytics 官方 | 非 YOLO 对照 |

**执行方式**：
```bash
# 统一数据集：stage4_full_tile_sensors3
# 统一 imgsz=960, epochs=100, patience=20
# 统一评估：val + public hard + false_alarm_v2

# 服务器脚本示例
python tools/train_yolo_api.py \
  --model yolov8n.pt \
  --data datasets/stage4_full_tile_sensors3/data.yaml \
  --project runs_baselines \
  --name yolov8n_3cls_e100 \
  --epochs 100 --batch 32 --imgsz 960 --device 0,1 \
  --patience 20
```

#### 3.2 效率指标采集（预计 0.5 天）

**目标**：对所有模型统一报告 Params/FLOPs/FPS。

```python
# 使用 ultralytics 内置工具
from ultralytics import YOLO
model = YOLO("best.pt")
# 获取 Params 和 FLOPs
info = model.info(verbose=True)  # 包含 params, GFLOPs
# 测量 FPS
import time
for _ in range(100):
    model.predict(source=image, imgsz=960, verbose=False)
# 报告: Params(M), GFLOPs, FPS(img/s), 模型大小(MB)
```

| 需报告模型 | 来源 |
|-----------|------|
| V4 / S5 / D5 / D7 | 服务器已有 |
| No-P2 | 服务器已有 |
| YOLOv8n / YOLOv8s / YOLOv8m / YOLOv10m / RT-DETR-l | P0 新训练 |

#### 3.3 Per-Size AP 分解（预计 0.5 天）

**目标**：对关键模型报告 AP_small / AP_medium / AP_large。

```python
# 使用 ultralytics 验证
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.val(
    data="data.yaml",
    imgsz=960,
    split="val"
)
# results.box.maps 包含 per-class AP
# 需要从 COCO 评估中提取 per-size AP
# 方法: 在 val 集上运行 predict，然后用 pycocotools 计算 AP_s/m/l
```

需报告 per-size AP 的模型：V4, S5, D5, No-P2, YOLOv8m(标准)

### P1：强烈建议完成（显著提升论文质量）

#### 3.4 SAHI 切片推理消融（预计 1 天）

**目标**：对比 full image vs sliced inference 对小目标的影响。

```python
# 安装 SAHI
# pip install sahi

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",
    confidence_threshold=0.20,
    image_size=960,
)

# Full inference
result = get_sliced_prediction(
    image=image,
    detection_model=detection_model,
    slice_height=640, slice_width=640,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25,
    perform_standard_pred=True,  # 同时做 full image 推理
)
```

| 实验项 | 模型 | 推理模式 | 报告指标 |
|--------|------|---------|---------|
| Full inference | D5 | full image | tiny recall, FP/image, FPS |
| Sliced inference | D5 | 640×640 slice | tiny recall, FP/image, FPS |
| Full+Sliced | D5 | full + slice 融合 | tiny recall, FP/image, FPS |

#### 3.5 视频级评估（预计 1-2 天）

**目标**：对 D5 和 V4 在至少 5 段视频上报告时序指标。

| 指标 | 计算方式 |
|------|---------|
| FP/min | 每分钟误报框数 |
| 首次检测时间 | 火焰/烟雾首次被检测到的帧号 ÷ FPS |
| 漏检事件数 | 有火/烟但整段未检测到的事件 |
| 时序闪烁率 | 同一目标检测/消失切换次数 ÷ 总帧数 |

视频来源：`D:\Researching\Yolo\Yolo\docs\video` 下的 15 个已测视频。

#### 3.6 失败案例可视化（预计 0.5 天）

**目标**：为论文提供 FP/FN 可视化案例。

| 案例类型 | 来源 | 数量 |
|---------|------|:---:|
| FP: 烟雾误判火焰 | public hard + false_alarm_v2 | 3-5 例 |
| FP: 背景误判火/烟 | false_alarm_v2 | 3-5 例 |
| FN: 小火点漏检 | public hard tiny recall 失败案例 | 3-5 例 |
| 混淆矩阵 | val 集预测 vs 标签 | 3×3 矩阵 |

### P2：可选增强（锦上添花）

#### 3.7 第三组 Seed 验证

D5 (seed=0) 和 D7 (seed=1) 已完全一致。补 seed=2 可形成均值+方差。

#### 3.8 Grad-CAM 可视化

对 D5 和 V4 的检测结果做注意力热力图，展示模型关注区域的差异。

#### 3.9 跨数据集泛化测试

用 D5 在 D-Fire 或 FASDD_RS 上测试，验证跨数据集泛化能力。

---

## 四、论文实验表格设计（最终版）

### Table 1: 主结果对比表

| Model | Backbone | P2 | Params(M) | GFLOPs | FPS | mAP50 | mAP50-95 | Tiny Recall | FP/1000 |
|-------|----------|:--:|:---------:|:------:|:---:|------:|--------:|----------:|:-------:|
| YOLOv8n | YOLOv8n | ✗ | 3.2 | — | — | — | — | — | — |
| YOLOv8s | YOLOv8s | ✗ | 11.2 | — | — | — | — | — | — |
| YOLOv8m | YOLOv8m | ✗ | 25.9 | — | — | — | — | — | — |
| YOLOv10m | YOLOv10m | ✗ | 24.6 | — | — | — | — | — | — |
| RT-DETR-l | RT-DETR | ✗ | 32.0 | — | — | — | — | — | — |
| No-P2 (ours) | YOLOv8m | ✗ | 25.9 | — | — | 0.841 | 0.580 | 0.706 | 15 |
| V4 (ours) | YOLOv8m-P2 | ✓ | 25.0 | — | — | 0.844 | **0.584** | 0.709 | 9 |
| S5 (ours) | YOLOv8m-P2 | ✓ | 25.0 | — | — | 0.838 | 0.584 | 0.722 | 5 |
| **D5 (ours)** | **YOLOv8m-P2** | **✓** | **25.0** | **—** | **—** | **0.832** | **0.574** | **0.720** | **5** |

### Table 2: 消融实验表

| 消融编号 | P2 | HN Mining | imgsz | val mAP50-95 | tiny recall | FP/1000 | Δ |
|---------|:--:|:---------:|:-----:|:-----------:|:----------:|:-------:|:--:|
| A1 (baseline) | ✗ | ✗ | 960 | 0.580 | 0.706 | 15 | — |
| A2 (+P2) | ✓ | ✗ | 960 | 0.584 | 0.709 | 9 | +0.004 |
| A3 (+HN) | ✓ | ✓ | 960 | **0.574** | **0.720** | **5** | -0.010 |
| A4 (+1280) | ✓ | ✓ | 1280 | 0.483 | 0.715 | 6 | -0.091 |

### Table 3: 部署阈值策略表

| 操作点 | conf | Recall | Tiny Recall | FP/image | FP/1000 | 适用场景 |
|--------|:----:|------:|:----------:|:--------:|:-------:|---------|
| Recall-first | 0.15 | 0.880 | 0.749 | 0.610 | 7 | 早期预警 |
| Balanced | 0.20 | 0.870 | 0.720 | 0.520 | 5 | 常规巡检 |
| Conservative | 0.30 | 0.839 | 0.646 | 0.348 | 2 | 自动告警 |

### Table 4: 视频级评估表

| 视频 | 模型 | FP/min | 首次检测(s) | 漏检事件 | 闪烁率 |
|------|------|:------:|:---------:|:-------:|:-----:|
| video_1 | V4 | — | — | — | — |
| video_1 | D5 | — | — | — | — |
| ... | ... | ... | ... | ... | ... |

---

## 五、执行优先级与时间线

| 优先级 | 任务 | 预计耗时 | 依赖 | 产出 |
|:---:|------|:---:|------|------|
| **P0-1** | 外部基线训练 (5模型) | 2-3天 | 服务器空闲 | Table 1 主表 |
| **P0-2** | 效率指标采集 | 0.5天 | P0-1 完成 | Params/FLOPs/FPS 列 |
| **P0-3** | Per-size AP 分解 | 0.5天 | 无 | AP_s/m/l 数据 |
| **P1-1** | SAHI 切片推理消融 | 1天 | 无 | Table 2 消融行 |
| **P1-2** | 视频级评估 | 1-2天 | 无 | Table 4 |
| **P1-3** | 失败案例可视化 | 0.5天 | 无 | 论文 Figure |
| P2-1 | 第三组 seed | 0.5天 | 无 | 统计可靠性 |
| P2-2 | Grad-CAM | 0.5天 | 无 | 可解释性 Figure |
| P2-3 | 跨数据集泛化 | 1天 | 无 | 泛化性证据 |

**关键路径**：P0-1 (外部基线) → P0-2 (效率指标) → P0-3 (per-size AP) → 论文初稿可写

**总预估**：P0 全部完成需 3-4 天，P0+P1 完成需 6-8 天。

---

## 六、论文贡献表述（基于已有实验）

### 6.1 可写入论文的贡献

1. **P2 小目标检测头消融**：tiny fire recall 0.709 vs 0.706（No-P2），holdout tiny fire Recall50 0.910 vs 0.836
2. **Hard negative mining 误报抑制**：false_alarm_v2 FP/1000 从 V4 的 9 降到 D5 的 5（-44%）
3. **多目标模型选择协议**：不只用 mAP，同时考虑 tiny recall + FP/image + FP/1000
4. **固定误报评估集**：1,000 张 verified-empty 图像，可复现
5. **部署阈值策略**：recall-first / balanced / conservative 三操作点
6. **可重复性验证**：D5/D7 双 seed 完全一致
7. **高分辨率负向消融**：imgsz 960→1280 导致 mAP 暴跌 0.091

### 6.2 暂不可写入的结论

- ❌ "我们的模型在 mAP 上显著优于所有基线"（当前 D5 mAP < V4）
- ❌ "SAHI 切片推理提升了小目标召回"（实验未做）
- ❌ "模型在视频场景下误报率低"（视频评估未做）

### 6.3 推荐的论文主线

```
面向无人机早期火灾预警的小目标检测与误报抑制

问题: UAV 场景小火/小烟检测困难 + 空场景误报严重
方法: P2 检测头 + Hard negative mining + 多目标评估协议
验证: 6 轮训练消融 + 1000 图误报集 + 多 seed 可重复性
部署: 三阈值操作点 + FPS/Params 效率报告
```

这个主线与我们已有数据完全一致，且不需要声称 mAP 突破。
