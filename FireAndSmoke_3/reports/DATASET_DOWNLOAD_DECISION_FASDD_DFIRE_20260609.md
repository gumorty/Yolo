# FASDD 与 D-Fire 下载决策和训练前处理规范

生成日期：2026-06-09

## 1. 是否要下载

结论：要下载，但不能直接全量混入训练。

优先级：

1. 先下载 `FASDD_UAV.zip`。
2. 再下载 D-Fire 的预划分版本或 Kaggle ready 版本。
3. 暂时不要先下载完整 FASDD 全量包进入训练。FASDD 数据总体约 76.49 GB，包含 CV、UAV、RS、SWIR 等部分；我们当前目标是无人机视角小火/小烟，所以第一阶段只用 UAV 子集最合理。

## 2. FASDD 为什么值得下载

Science Data Bank 页面显示，FASDD 是 Flame And Smoke Detection Dataset，面向 object detection，包含 fire、smoke 和 confusing non-fire/non-smoke 图像，来源覆盖 surveillance cameras、UAVs 和 satellites。它由三个子集组成：

| 子集 | 数量/特点 | 对我们的价值 |
|---|---|---|
| FASDD_CV | 95,314 samples | 普通视觉火/烟，域较杂 |
| FASDD_UAV | 25,097 samples，UAV captured | 当前最优先，贴近无人机视角 |
| FASDD_RS | 2,223 samples，satellite imagery | 更偏遥感卫星，不适合作为当前主训练 |

FASDD 全集包含 122,634 samples，其中 70,581 positive、52,073 negative；共有 113,154 fire instances 和 73,072 smoke instances。FASDD_UAV 包含 36,308 fire instances 和 17,222 smoke instances。标注格式包含 YOLO、VOC、COCO、TDML，类别为 `fire` 和 `smoke`。这正好补足我们第三轮缺少小烟/小火公开 UAV 数据的问题。

推荐使用方式：

- 第一阶段只用 FASDD_UAV。
- 不直接用 FASDD_RS，因为遥感卫星图和无人机巡检视频尺度、纹理、视角差别较大。
- FASDD_CV 可作为后续补充，但不能压过 UAV 数据。

## 3. D-Fire 为什么值得下载

D-Fire GitHub 页面显示，它是 fire/smoke object detection 数据集，超过 21,000 images，YOLO 标注格式。类别统计：

| 场景 | 图像数 |
|---|---:|
| only fire | 1,164 |
| only smoke | 5,867 |
| fire and smoke | 4,658 |
| none | 9,838 |

目标框统计：

| 类别 | bbox 数 |
|---|---:|
| fire | 14,692 |
| smoke | 11,865 |

D-Fire 的优势是 smoke-only 和 none 负样本很多，能补我们当前 `small smoke` 和负样本多样性。但它主要不是 UAV 视角，所以只能受控混入。

推荐使用方式：

- 下载预划分 train/val/test 或 Kaggle ready 版本。
- 训练集中先限制 D-Fire train 最多 5,000 张，val 最多 1,000 张。
- 如果 readiness 仍然显示 small smoke 不够，再逐步增加。
- 不要让 D-Fire 占比超过 stage4 训练集的 30%-35%，否则会把模型拉向地面/监控视角。

## 4. 下载后目录规范

放到：

```text
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\downloads\
  FASDD_UAV\
  DFire\
```

规范化后输出到：

```text
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_sources\
  fasdd_uav_curated\
  dfire_curated\
```

每个 curated 数据集必须有：

```text
train/images
train/labels
val/images
val/labels
test/images
test/labels
data.yaml
```

最终类别统一为：

```text
0 fire
1 other
2 smoke
```

FASDD/D-Fire 原始两类通常是：

```text
0 fire
1 smoke
```

因此导入时要映射：

```text
0 -> 0 fire
1 -> 2 smoke
```

## 5. 规范化命令

如果下载解压后的目录已经是 YOLO 风格，例如：

```text
FASDD_UAV/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
```

运行：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\17_prepare_stage4_source_yolo.ps1 `
  -Root D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\downloads\FASDD_UAV `
  -Out D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_sources\fasdd_uav_curated `
  -SourceId fasdd_uav `
  -ClassMap 0=0,1=2 `
  -Overwrite
```

D-Fire：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\17_prepare_stage4_source_yolo.ps1 `
  -Root D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\downloads\DFire `
  -Out D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_sources\dfire_curated `
  -SourceId dfire `
  -ClassMap 0=0,1=2 `
  -IncludeEmpty `
  -Overwrite
```

`-IncludeEmpty` 对 D-Fire 很重要，因为它有大量 none 图像，这些图像能当负样本。FASDD 是否使用空负样本，需要看其解压结构和审计结果。

## 6. 严格筛选和审计规则

规范化后立刻审计：

```powershell
python D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\tools\dataset_audit.py `
  --data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_sources\fasdd_uav_curated\data.yaml `
  --out D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\reports\fasdd_uav_curated.audit.json
```

D-Fire 同理。

必须检查：

- 是否有缺失 label；
- 是否有大量空 label；
- `class_area_bins` 中 `smoke|tiny/small` 是否足够；
- `image_shapes` 中是否有大量高分辨率图；
- fire/smoke 是否类别映射正确；
- 随机抽样可视化 100 张，人工确认 bbox 没有错位。

## 7. 合并策略

编辑：

```text
configs/stage4_dataset_mix_template.yaml
```

将 FASDD_UAV 和 D-Fire 打开：

```yaml
fasdd_uav_curated:
  enabled: true
dfire_curated:
  enabled: true
```

D-Fire 保持限制：

```yaml
max_images:
  train: 5000
  val: 1000
```

然后运行：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\14_build_stage4_mixed_dataset.ps1 -Overwrite
```

## 8. full+tile 构建

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\11_build_highres_tiles.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_mixed_3cls\data.yaml `
  -Out D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3 `
  -CopyFull `
  -Overwrite
```

这一步是吸收 SAHI 经验的关键：切片不只放在推理端，而是进入训练数据。它对高分辨率 UAV 图才有意义。

## 9. readiness 检查

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\16_check_stage4_readiness.ps1
```

如果失败，不训练。失败项通常说明：

- small smoke 不够；
- 高分辨率图不够；
- hard negative 不够；
- val 中没有足够小目标，无法可信评估。

## 10. 是否直接开训

不能直接 120 轮。

下载 FASDD_UAV 和 D-Fire 后，应按下面流程：

```text
下载
-> 规范化
-> 单数据集审计
-> 合并
-> full+tile
-> readiness PASS
-> 20 epoch scout
-> holdout 测试
-> 达标再 120 epoch
```

这是针对第三轮失败最直接的修正：先让数据具备“小火/小烟/高分辨率/UAV 视角”的先验，再让模型学习。

## 11. 来源

- FASDD / Science Data Bank: https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda
- FASDD paper DOI: https://doi.org/10.1080/10095020.2024.2347922
- D-Fire GitHub: https://github.com/gaia-solutions-on-demand/DFireDataset
