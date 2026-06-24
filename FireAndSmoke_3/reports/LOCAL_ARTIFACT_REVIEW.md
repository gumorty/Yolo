# 第三轮训练前本地实验复盘

## 已有模型结果

从 `D:\Researching\Training_After` 中读取到的最终指标如下：

| 模型 | epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8m Sensors 2类 | 100 | 0.8539 | 0.8325 | 0.8905 | 0.6149 |
| YOLOv8m Sensors 3类 | 100 | 0.8360 | 0.7805 | 0.8456 | 0.5736 |
| YOLOv10m Sensors 3类 | 100 | 0.8274 | 0.7801 | 0.8463 | 0.5771 |
| YOLOv8m Sensors+自有 2类 | 150 | 0.8715 | 0.8207 | 0.8856 | 0.6106 |
| YOLOv8m Sensors+自有 3类 | 150 | 0.8374 | 0.7798 | 0.8391 | 0.5715 |

结论：第二轮融合自有数据后，2类模型 Precision 略升，但 Recall、mAP50 和 mAP50-95 没有超过第一轮 Sensors 2类；3类模型也没有超过第一轮 Sensors 3类。这说明“数据更多”没有自动带来“检测更全”，继续盲目混合数据或直接堆到 300 epoch 风险很高。

## 主要问题定位

1. 数据源标注风格不一致。Sensors 数据中 fire/smoke/other 的定义更完整，自有 fire_pic 数据可能更偏局部或粗框，混合后模型会学习到不稳定的框选习惯。
2. 第二轮没有同步增强 other。自有数据主要增加 fire/smoke，3类模型中的 other hard negative 没有等比例增加，所以灯光、反光、车灯等误报抑制没有明显增强。
3. 小目标不是单纯训练轮次问题。无人机远景中的小火苗缩放到 640 后特征很弱，需要 P2 检测头、高分辨率训练/推理或切片推理。
4. 早期烟雾不是单帧 YOLO 的强项。弱烟低对比、边界模糊，需要低阈值候选、连续帧确认，进一步还需要热红外或时序信息。

## 第三轮原则

第三轮不默认使用第二轮融合权重作为主线初始模型，而优先使用第一轮 Sensors best.pt：

- 3类主线：`sensors_yolov8m_3cls_100_best.pt`
- 2类对照：`sensors_yolov8m_2cls_100_best.pt`

自有数据暂时不直接混入主训练集，先作为单独审计对象和 hard-case 候选来源。只有在确认标注规则一致、other 类补齐、外部视频测试能收益后，再做小规模高质量微调。

## 本轮新增数据审计发现

本地审计脚本已经生成：

- `reports/strict_sensors_3cls.audit.json`
- `reports/strict_sensors_2cls.audit.json`
- `reports/self_fire_pic_2cls_standalone.audit.json`

关键发现：

| 数据集 | 图像数 | fire | smoke | other | tiny | small |
|---|---:|---:|---:|---:|---:|---:|
| Sensors 3类 | 27294 | 26717 | 24560 | 14671 | 715 | 17334 |
| Sensors 2类 | 27294 | 26717 | 24560 | 0 | 79 | 8374 |
| 自有 fire_pic 2类 | 16039 | 29137 | 12958 | 0 | 4121 | 13083 |

这说明自有 fire_pic 并不是没价值。相反，它的小目标样本很多，适合用于提升微小 fire/smoke 召回。但它的问题是没有 other 类；如果直接合并到 3类训练，会增加 fire/smoke 占比，却不增加 hard negative，容易削弱 3类模型对灯光、反光、车灯等干扰的判断边界。

因此第三轮推荐顺序是：

1. 先训练 Sensors-only P2 模型，验证 P2 对小目标是否有效。
2. 再把自有 fire_pic 作为 2类小目标增强对照，而不是立刻混到 3类主线。
3. 如果要做 3类增强，必须同步补充 other hard negative，否则会重复第二轮问题。
