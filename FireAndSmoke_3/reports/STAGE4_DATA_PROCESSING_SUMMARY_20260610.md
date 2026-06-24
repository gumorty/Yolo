# Stage4 数据处理执行总结

生成日期：2026-06-10

## 1. 本次处理目标

第三轮 YOLOv8m-P2 训练没有达到预期，主要原因是训练数据仍以 `640x640` 图像为主，且 tiny/small smoke 极少。Stage4 的目标是先修复数据先验，再进行短训验证，最后才决定是否进行 120 轮长训。

本次已完成：

1. 读取并识别下载后的 `FASDD_UAV` 和 `D-Fire` 真实目录结构；
2. 统一转换为三类 YOLO 格式：`fire / other / smoke`；
3. 单数据集审计；
4. 构建公开 hard holdout；
5. 构建 Stage4 mixed 数据集；
6. 构建 full+tile 训练数据集；
7. 进行 readiness PASS/FAIL 检查；
8. 生成随机标注可视化拼图。

## 2. 原始数据结构识别

### FASDD_UAV

真实结构：

```text
FASDD_UAV/
  images/
  annotations/YOLO_UAV/
    train.txt
    val.txt
    test.txt
    labels/
```

split 数量：

| split | images |
|---|---:|
| train | 12,551 |
| val | 8,365 |
| test | 4,181 |

图像分辨率主要为 `>1280` 或 `961-1280`，符合无人机高分辨率小目标训练需求。

### D-Fire

真实结构：

```text
D-Fire/
  train/images
  train/labels
  test/images
  test/labels
```

数量：

| split | images |
|---|---:|
| train | 17,221 |
| test | 4,306 |

由于 D-Fire 没有 val，本次将 test 同步复制为 val 候选，用于受控混入验证。

## 3. 规范化输出

### FASDD_UAV curated

输出：

```text
datasets/stage4_sources/fasdd_uav_curated/data.yaml
```

转换结果：

| split | copied | empty labels | objects | missing labels |
|---|---:|---:|---:|---:|
| train | 12,551 | 5,994 | 26,776 | 0 |
| val | 8,365 | 3,995 | 17,828 | 0 |
| test | 4,181 | 1,997 | 8,926 | 0 |

### D-Fire curated

输出：

```text
datasets/stage4_sources/dfire_curated/data.yaml
```

转换结果：

| split | copied | empty labels | objects | missing labels |
|---|---:|---:|---:|---:|
| train | 17,221 | 7,833 | 21,364 | 0 |
| test | 4,306 | 2,005 | 5,193 | 0 |
| val_from_test | 4,306 | 2,005 | 5,193 | 0 |

类别映射：

```text
原始 0 -> fire 0
原始 1 -> smoke 2
other 1 保留给 Sensors 和 hard negative
```

## 4. 单数据集审计结果

### FASDD_UAV

关键结果：

| 指标 | 数值 |
|---|---:|
| images | 25,097 |
| fire instances | 36,308 |
| smoke instances | 17,222 |
| train tiny+small fire | 13,364 |
| train tiny+small smoke | 1,706 |
| val tiny+small fire | 8,976 |
| val tiny+small smoke | 1,092 |
| high-res images | 25,097 |

FASDD_UAV 是本轮最关键的数据补充，它直接解决了第三轮缺少 UAV 高分辨率小目标样本的问题。

### D-Fire

关键结果：

| 指标 | 数值 |
|---|---:|
| images | 25,833 |
| fire instances | 14,164 |
| smoke instances | 17,570 |
| train tiny+small fire | 1,896 |
| train tiny+small smoke | 7,601 |
| valid tiny+small fire | 499 |
| valid tiny+small smoke | 1,860 |

D-Fire 对 smoke 尤其有价值，但不是 UAV 主场景，所以本次只受控混入：train 最多 5,000 张，val 最多 1,000 张。

## 5. Stage4 mixed 数据集

输出：

```text
datasets/stage4_mixed_3cls/data.yaml
```

组成：

- Sensors 3类：全部保留；
- FASDD_UAV：train/val 全部保留，包括空负样本；
- D-Fire：train 5,000，val 1,000，保留空负样本；
- FASDD_UAV test 不进入 mixed 训练，用于 holdout。

审计结果：

| split | images | empty labels |
|---|---:|---:|
| train | 34,856 | 8,819 |
| val | 16,613 | 4,655 |
| test | 2,741 | 47 |

关键 readiness 指标：

| 检查项 | 数值 | 门槛 | 结果 |
|---|---:|---:|---|
| train tiny+small fire | 19,295 | 5,000 | PASS |
| train tiny+small smoke | 4,063 | 1,000 | PASS |
| valid tiny+small fire | 11,088 | 300 | PASS |
| valid tiny+small smoke | 1,533 | 100 | PASS |
| total tiny+small other | 9,596 | 1,000 | PASS |
| high-res ratio | 0.4584 | 0.2 | PASS |

结论：mixed 数据先验已通过。

## 6. Public hard holdout

输出：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

构建方式：

- 只从 FASDD_UAV test split 中抽取；
- 不参与 Stage4 mixed train/val；
- 500 张固定评估图；
- 包含 100 张空负样本。

审计结果，按唯一 500 张计算：

| 指标 | 数值 |
|---|---:|
| images | 500 |
| empty labels | 100 |
| fire instances | 1,278 |
| smoke instances | 535 |
| tiny+small fire | 977 |
| tiny+small smoke | 126 |
| high-res images | 500 |

结论：可以作为公开 hard holdout，用于 Stage4 scout 与旧模型对比。

## 7. Full+tile 数据集

输出：

```text
datasets/stage4_full_tile_sensors3/data.yaml
```

构建参数：

```text
tile_size = 960
overlap = 0.25
min_visibility = 0.35
negative_ratio = 0.05
copy_full = true
```

构建结果：

| split | source full | positive tiles | negative tiles |
|---|---:|---:|---:|
| train | 34,856 | 37,087 | 3,801 |
| val | 16,613 | 19,231 | 2,462 |
| test | 2,741 | 2,513 | 10 |

最终审计：

| split | images | empty labels |
|---|---:|---:|
| train | 75,744 | 12,560 |
| val | 38,306 | 7,099 |
| test | 5,264 | 47 |

最终 readiness：

| 检查项 | 数值 | 门槛 | 结果 |
|---|---:|---:|---|
| train tiny+small fire | 47,119 | 5,000 | PASS |
| train tiny+small smoke | 9,011 | 1,000 | PASS |
| valid tiny+small fire | 27,793 | 300 | PASS |
| valid tiny+small smoke | 3,290 | 100 | PASS |
| total tiny+small other | 15,792 | 1,000 | PASS |
| high-res ratio | 0.5365 | 0.2 | PASS |

结论：full+tile 数据先验已通过，可以进入 10-20 epoch scout training。

## 8. 可视化检查

已生成：

```text
reports/fasdd_uav_label_contact_sheet.jpg
reports/dfire_label_contact_sheet.jpg
reports/stage4_full_tile_label_contact_sheet.jpg
```

抽样结果显示：

- fire 红框；
- smoke 蓝框；
- other 绿框；
- 类别映射没有明显错位；
- FASDD_UAV 中 UAV 高分辨率小火点和大烟框都存在；
- D-Fire 中 smoke-only 和 none/hard negative 价值明显；
- full+tile 中能看到小目标局部被有效放大。

## 9. 下一步是否允许训练

允许进入 scout training。

但不允许直接 120 轮长训。下一步应先运行：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\12_train_stage4_scout.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3\data.yaml `
  -Name stage4_scout_p2_fasdd_dfire_tile_e20
```

服务器 Linux：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 \
DATA=$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml \
NAME=stage4_scout_p2_fasdd_dfire_tile_e20 \
bash scripts_linux/12_train_stage4_scout.sh
```

scout 训练后必须在下面数据上对比第一轮旧模型和第三轮 P2 模型：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

进入 120 轮长训的条件：

1. public hard holdout 的 small fire/smoke recall 明显高于第一轮；
2. overall recall 不下降；
3. 误报数量可控；
4. 视频首次检测时间不变差；
5. GPU FPS 可接受。

如果 scout 没有提升，不继续长训，回到数据比例和 tile 策略调整。
