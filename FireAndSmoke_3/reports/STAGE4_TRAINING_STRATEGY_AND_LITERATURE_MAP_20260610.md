# Stage4 训练策略与文献经验映射

生成日期：2026-06-10

## 1. 第四轮不是只加数据，而是换训练策略

第一轮、第二轮基本沿用原论文策略；第三轮只增加了 P2 检测头和 `imgsz=960`，但结果没有达到预期。核心原因是：模型结构改变了，训练数据和选择指标没有同步改变。P2 小目标检测头没有足够 tiny/small smoke 和高分辨率 UAV 监督，最终只带来更慢的推理和更保守的检测。

第四轮已经加入的策略不是“堆模块”，而是围绕论文目标构建一个闭环：

```text
数据先验达标
-> full+tile 小目标训练
-> P2 重新验证
-> smoke-safe augmentation
-> hard negative 控误报
-> 20轮 scout
-> public hard holdout 选择模型
-> 通过后再长训
```

## 2. 已经采用的高效策略

### 2.1 数据先验门控

痛点：第三轮训练前没有强制检查 tiny/small smoke 和高分辨率比例，导致 P2 在不合适的数据上训练。

做法：新增 readiness 检查，只有以下条件满足才允许进入 scout：

| 检查项 | Stage4 当前值 | 门槛 |
|---|---:|---:|
| train tiny+small fire | 47,119 | 5,000 |
| train tiny+small smoke | 9,011 | 1,000 |
| valid tiny+small fire | 27,793 | 300 |
| valid tiny+small smoke | 3,290 | 100 |
| total tiny+small other | 15,792 | 1,000 |
| high-res ratio | 0.5365 | 0.2 |

影响：保证训练前数据已经对准“小火/小烟/高分辨率 UAV 场景”，不再让结构替数据背锅。

### 2.2 FASDD_UAV + D-Fire 受控混合

痛点：当前 Sensors 数据无法支撑 UAV 小目标早期预警，尤其缺少 small smoke。

做法：

- FASDD_UAV 作为 UAV 火/烟主数据；
- D-Fire 只受控混入，补 smoke-only 和负样本；
- Sensors 保留历史基线分布；
- FASDD_UAV test 单独构建 public hard holdout，不进训练。

影响：模型不再只学习 640 图和明显火/烟，同时避免 D-Fire 过度拉向地面/监控视角。

### 2.3 Full + Tile Training

文献依据：SAHI 与 YOLO sliced inference/fine-tuning 相关工作指出，小目标在整图缩放后信息不足，切片推理和切片微调组合能提高小目标表现。

痛点：高分辨率大图里的远处小火点直接缩放到 960 后仍然很小。

做法：

```text
tile_size = 960
overlap = 0.25
min_visibility = 0.35
negative_ratio = 0.05
copy_full = true
```

输出：

| split | full images | positive tiles | negative tiles |
|---|---:|---:|---:|
| train | 34,856 | 37,087 | 3,801 |
| val | 16,613 | 19,231 | 2,462 |
| test | 2,741 | 2,513 | 10 |

影响：full image 保留上下文，tile 放大小目标细节，这是第四轮最关键的训练策略之一。

### 2.4 P2 检测头重新验证

文献依据：SOD-YOLO、YOLO-TLA、LAF-YOLOv10 等小目标检测工作都使用额外浅层检测头或高分辨率检测层解决小目标空间信息丢失。

痛点：第三轮 P2 没提升，但原因不是 P2 必然无效，而是数据不支持。

做法：第四轮仍保留 YOLOv8m-P2，但只在数据先验 PASS 后测试。这样 P2 的成败才有解释力。

影响：如果 scout 提升，就说明 P2 + 高分辨率 tile 数据有效；如果仍不提升，则下一步应对比普通 YOLOv8m，而不是继续盲目加结构。

### 2.5 Smoke-safe Augmentation

痛点：烟雾是弱边界、弱纹理目标。过强的 erasing、mixup、copy-paste、randaugment 可能破坏烟雾形态。

第四轮 scout 设置：

```text
mosaic = 0.20
mixup = 0.0
copy_paste = 0.0
erasing = 0.0
auto_augment = none
close_mosaic = 5
```

影响：减少对烟雾纹理的破坏，让模型更稳定学习微弱烟雾和远景烟雾。

### 2.6 Hard Negative 与空负样本保留

文献经验：早期预警系统不能只提高召回，还要控制误报。Infra-YOLO 等红外小目标工作也强调背景噪声抑制和特征融合对降低误报的重要性。

痛点：如果只补 fire/smoke，模型容易把云雾、反光、灯光、灰尘误报成火/烟。

做法：

- 保留 Sensors 的 `other` 类；
- 保留 FASDD_UAV 空负样本；
- 保留 D-Fire none 负样本；
- tile 数据中保留 5% negative tiles。

影响：模型在偏召回的同时仍有背景抑制信号。

### 2.7 Scout Before Long Training

痛点：120 轮训练成本高，第三轮已经证明长训不等于提升。

做法：

- 先 20 轮 scout；
- 每 5 轮保存一次；
- 比较 epoch 5/10/15/20、best.pt、last.pt；
- 只在 public hard holdout 上提升后再长训。

影响：用较小成本验证数据和策略是否有效，避免再次昂贵失败。

### 2.8 Hard Holdout Model Selection

文献经验：SmokeyNet 一类早期烟雾检测工作强调 time-to-detection，而不是只看单帧分类指标。对我们的任务来说，也不能只看 mAP50-95。

痛点：第三轮 best.pt 按默认 mAP50-95 保存，但我们的目标是小目标召回和视频预警提前。

做法：新增 public hard holdout：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

模型选择重点：

- small fire recall；
- small smoke recall；
- false positives；
- first detection time；
- FPS。

影响：模型选择和论文目标一致。

## 3. 文献中有价值但暂不直接加入的策略

### 3.1 Cross-scale / Cross-layer Neck

CF-YOLO 提出 CS-FPN、FRM、Sandwich Fusion、LSDECD；CFPT 提出跨层通道/空间注意力，目标都是缓解 FPN 融合过程中的语义差距、空间细节损失和位置偏差。

为什么暂缓：这些属于模型结构级改造。如果和数据、tile、P2 同时加入，无法解释提升来自哪里。

下一步：如果 Stage4 scout 有提升但仍漏小烟，再做单独 neck ablation。

### 3.2 DySample / Attention-guided FPN

LAF-YOLOv10 一类工作用 P2、注意力引导 FPN、DySample、Wise-IoU 等组合改善无人机小目标。

为什么暂缓：DySample 和 Wise-IoU 需要改 Ultralytics 模块或 loss，风险高于当前数据策略。

下一步：作为 Stage5 结构消融候选，而不是 Stage4 首轮混入。

### 3.3 Soft-NMS / WBF

SOD-YOLO 使用 Soft-NMS 以保留密集小目标真阳性。

为什么有价值：多火点密集区域中，标准 NMS 可能压掉邻近火点。

下一步：这可以先作为推理策略测试，不一定进入训练。scout 后如果多火点漏检明显，再加入 Soft-NMS/WBF 对比。

### 3.4 Distillation / Pruning / TensorRT

DAMO-YOLO、Infra-YOLO 都强调效率、压缩和部署。

为什么暂缓：目前准确率和召回还没解决，先做压缩会让目标混乱。

下一步：等 Stage4 训练出更强 teacher，再做蒸馏、剪枝、ONNX/TensorRT。

## 4. 为什么不能“全部学习并全部加入”

论文里的方法通常解决不同痛点：

| 方法 | 解决问题 | 当前是否采用 |
|---|---|---|
| P2 小目标头 | 浅层空间细节丢失 | 已采用 |
| sliced fine-tuning | 大图小目标信息不足 | 已采用 |
| hard negative | 误报控制 | 已采用 |
| smoke-safe augmentation | 烟雾纹理破坏 | 已采用 |
| hard holdout | 指标与目标不一致 | 已采用 |
| CS-FPN/CFPT | 跨层融合不足 | 暂缓，后续消融 |
| DySample/attention FPN | 上采样和融合偏差 | 暂缓，后续消融 |
| Wise-IoU/NWD | 小框定位敏感 | 暂缓，后续消融 |
| Soft-NMS/WBF | 密集多火点 NMS 抑制 | scout 后推理测试 |
| Distillation/pruning | 部署速度 | 准确率达标后做 |

如果一次性全部加入，训练结果即使提升，也无法证明是哪个策略有效；如果不提升，也不知道哪一部分拖累了模型。

## 5. Stage4 当前推荐训练命令

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\12_train_stage4_scout.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3\data.yaml `
  -Name stage4_scout_p2_fasdd_dfire_tile_e20
```

服务器：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 \
DATA=$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml \
NAME=stage4_scout_p2_fasdd_dfire_tile_e20 \
bash scripts_linux/12_train_stage4_scout.sh
```

## 6. Stage4 成功判定

Stage4 scout 成功不是总 mAP 暴涨，而是：

1. public hard holdout 上 small fire/smoke recall 明显超过第一轮；
2. 视频首次检测时间提前或不变差；
3. 误报数量可通过连续帧确认控制；
4. FPS 在 GPU 环境下可接受；
5. 若 P2 不提升，则用相同数据跑普通 YOLOv8m scout 做对照。

## 7. 参考来源

- Akyon et al., SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection, arXiv:2202.06934.
- Keles et al., Evaluation of YOLO Models with Sliced Inference for Small Object Detection, arXiv:2203.04799.
- Wang et al., CF-YOLO for small target detection in drone imagery based on YOLOv11 algorithm, Scientific Reports 2025, DOI: 10.1038/s41598-025-99634-0.
- Du et al., Cross-Layer Feature Pyramid Transformer for Small Object Detection in Aerial Images, arXiv:2407.19696.
- Wang and Zhao, SOD-YOLO: Enhancing YOLO-Based Detection of Small Objects in UAV Imagery, arXiv:2507.12727.
- Chen et al., Infra-YOLO: Efficient Neural Network Structure with Model Compression for Real-Time Infrared Small Object Detection, arXiv:2408.07455.
- Xu et al., DAMO-YOLO: A Report on Real-Time Object Detection Design, arXiv:2211.15444.
