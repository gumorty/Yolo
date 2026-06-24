# Stage4 下一步项目修改与再训练执行清单

生成日期：2026-06-09

## 1. 当前目标

下一轮训练的目标不是让总 mAP 轻微波动，而是让论文主张能够成立：

- 无人机大场景中的小火苗、小烟雾、多火点召回提高；
- 视频首次检测时间提前或至少不变差；
- 误报能通过连续帧确认、阈值和 hard negative 控制；
- 速度在 GPU 环境下可接受；
- 高温潜在危险源作为独立热成像风险分支规划，不再混到纯 RGB 检测结论里。

第三轮结果已经说明：只加 P2 和 `imgsz=960` 不够。下一轮必须先改数据和评估，再改训练。

## 2. 项目需要改成什么样

### 2.1 数据结构

建议新增或整理成下面结构：

```text
datasets/
  stage4_sources/
    uav_hard_raw/
      images/
      labels/
    dfire_curated/
      images/
      labels/
    fasdd_uav_curated/
      images/
      labels/
    hard_negative/
      images/
      labels/
  stage4_full_tile_sensors3/
    train/
    val/
    test/
    data.yaml
```

其中 `uav_hard_raw` 和最终的 `uav_hard_eval` 必须严格分开。hard eval 只用于测试，不参与训练。

### 2.2 类别设计

短期继续使用三类：

```text
0 fire
1 other
2 smoke
```

原因是 fire/smoke 的误报干扰很多，比如灯光、太阳反光、云雾、灰尘、车灯、白色水汽。如果下一轮只做二类，召回可能上升，但误报会更难压。三类路线更适合论文里的预警系统。

但 `other` 不能无限扩充。它要作为 hard negative，不是普通背景类。建议标注：

- 类火亮点；
- 类烟云雾；
- 灰尘/水汽/雾气；
- 车灯/路灯/太阳反光；
- 红色物体和高亮区域。

### 2.3 训练数据构建

使用新脚本从高分辨率图像生成 full+tile 数据：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\11_build_highres_tiles.ps1 -CopyFull -Overwrite
```

服务器：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 COPY_FULL=1 OVERWRITE=1 bash scripts_linux/11_build_highres_tiles.sh
```

关键点：

- 如果源图仍然是 640x640，这一步没有根本收益；
- 必须尽量使用原始无人机视频抽帧或外部高分辨率 UAV 火/烟数据；
- tile size 先用 960，overlap 先用 0.25；
- 只保留包含 fire/smoke 的正样本 tile 和少量 hard negative tile；
- full image 保留全局上下文，tile 负责放大小目标。

### 2.4 数据审计

构建后必须跑：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\13_audit_stage4_dataset.ps1
```

服务器：

```bash
bash scripts_linux/13_audit_stage4_dataset.sh
```

重点看输出里的：

- `image_shapes`：是否真的有 960 以上或 1280 以上图像来源；
- `class_area_bins`：fire/smoke 的 tiny 和 small 是否明显增加；
- `empty_labels`：负样本是否可控；
- `other|tiny/small`：hard negative 是否足够，但不能压过 fire/smoke。

建议进入训练前的最低门槛：

| 项目 | 最低要求 |
|---|---:|
| hard eval images | 300-500 |
| tiny+small fire | 至少当前 valid/test 的 3-5 倍 |
| small smoke | 至少当前 valid/test 的 5-10 倍 |
| high-res source images | 不应几乎全是 640 |
| hard negative | 与 fire/smoke 正样本成比例 |

## 3. 下一轮训练怎么跑

### 3.1 不直接跑 120 轮

先跑 20 轮 scout：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\12_train_stage4_scout.ps1
```

服务器：

```bash
bash scripts_linux/12_train_stage4_scout.sh
```

默认策略已经改成：

- P2 三类模型；
- 从第一轮 3 类 best.pt 迁移；
- `imgsz=960`；
- `mosaic=0.20`；
- `mixup=0`；
- `copy_paste=0`；
- `erasing=0`；
- 每 5 轮保存一次权重；
- patience=8。

这样做的原因是烟雾是弱边界、弱纹理目标，强 RandAugment、erasing、mixup 很容易破坏烟雾形态。第三轮默认增强太强，可能让模型更保守。

### 3.2 scout 后如何选择模型

不要只看 Ultralytics 默认 `best.pt`。要比较：

- epoch 5；
- epoch 10；
- epoch 15；
- epoch 20；
- `best.pt`；
- `last.pt`。

对每个权重都在固定 hard set 上评估：

- small fire recall；
- small smoke recall；
- overall recall；
- precision-like；
- false positives per minute；
- first detection time；
- GPU FPS。

如果 epoch 10 的 small recall 最好，哪怕 `best.pt` mAP50-95 更高，也应该保留 epoch 10 作为论文候选模型。这一点要和第三轮区分开。

## 4. 数据集选择顺序

优先级如下：

1. 自有无人机视频抽帧并标注，尤其是失败案例和远景小火/小烟。
2. FASDD-UAV，如果能下载并确认许可。
3. D-Fire，受控混入，主要补 fire/smoke 多样性和 none 负样本。
4. VisDrone，只用于可选 UAV 小目标预训练，不直接当火/烟训练集。
5. TinyPerson，只作为小目标尺度研究参考，短期不建议投入训练资源。
6. HIT-UAV 和 FLAME，放到热成像风险分支，不进入当前 RGB YOLO 三类主线。

## 5. 是否需要换模型结构

短期不建议马上大改成 YOLOv11/CF-YOLO 复刻版。原因是现在最大的瓶颈是数据和评估，不是已经证明结构不足。

建议顺序：

1. 先用 YOLOv8m 和 YOLOv8m-P2 在改好的数据上 scout；
2. 如果 full+tile 数据明显提升，再考虑保留 P2；
3. 如果 P2 仍然只提高误报控制、不提高召回，则回退普通 YOLOv8m；
4. 如果数据路线有效但小目标仍漏检，再做轻量 neck 改造，比如跨层特征融合、注意力重校准或 DySample；
5. 不要一次加入多个模块，否则论文消融无法解释。

## 6. 什么时候可以跑 120 轮

只有满足下面条件才跑：

| 条件 | 门槛 |
|---|---|
| hard set small fire/smoke recall | 比第一轮提升至少 5 个百分点 |
| overall recall | 不比第一轮低超过 1 个百分点 |
| 视频首次检测时间 | 不晚于第一轮，最好提前 |
| 误报/分钟 | 可通过连续帧确认压住 |
| GPU FPS | 在目标部署环境中可接受 |

如果 scout 没过，不要长训。继续补数据和修标注。

## 7. 论文实验预期如何改写

论文里不要承诺“总 mAP 大幅提升”。更合理的贡献表达是：

1. 面向无人机火烟预警的 hard-case 数据构建与评估协议；
2. full-frame + tile training 缓解大场景小目标信息丢失；
3. 三类 hard negative 设计降低类火/类烟误报；
4. 视频级连续帧确认提升预警稳定性；
5. 对潜在高温风险区，提出 RGB 检测与热成像风险分支的系统框架。

最理想的结果是：

- 总 mAP50 保持 0.84 左右或略升；
- mAP50-95 不下降；
- small fire/smoke recall 明显提升；
- 视频首次检测时间提前；
- 误报通过策略控制；
- GPU FPS 有清楚报告。

## 8. 立即执行顺序

1. 整理 300-500 张固定 UAV hard eval，不参与训练。
2. 抽取自有视频 hard frames，优先标远处小火、小烟、多火点和干扰物。
3. 获取并审计 FASDD-UAV / D-Fire，转换成统一三类标注。
4. 构建 stage4 full+tile 数据集。
5. 跑 stage4 数据审计，看 class×size 是否达标。
6. 跑 20 轮 scout。
7. 对每 5 轮权重做 hard-set 图片和视频评估。
8. 只有 scout 通过，再跑 120 轮正式训练。
