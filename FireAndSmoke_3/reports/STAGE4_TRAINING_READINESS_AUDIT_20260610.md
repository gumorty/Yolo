# Stage4 训练准备度最终审计

生成日期：2026-06-10

## 1. 结论

Stage4 当前已经准备充分，可以进入下一轮 `20 epoch scout training`。

不建议直接进入 120 轮长训。原因不是准备不足，而是我们本轮策略明确要求：先用较低成本验证数据重构、P2、full+tile 和 smoke-safe augmentation 是否真的提升 public hard holdout 的 small fire/smoke recall。只有 scout 达标后，才进入 120 轮正式训练。

## 2. 已通过的关键检查

预检查报告：

```text
reports/stage4_preflight_check.json
```

结果：`PASS`

通过项：

| 检查项 | 结果 |
|---|---|
| Stage4 full+tile data.yaml 存在 | PASS |
| public hard holdout data.yaml 存在 | PASS |
| YOLOv8m-P2 3类模型 YAML 存在 | PASS |
| 第一轮 3类迁移权重存在 | PASS |
| data.yaml 使用相对路径 | PASS |
| holdout data.yaml 使用相对路径 | PASS |
| train images 数量 | 75,744 |
| val images 数量 | 38,306 |
| holdout images 数量 | 500 |
| 模型 YAML 可构建 | PASS |
| 迁移权重可加载 | PASS |

模型加载检查显示：

```text
Transferred 319/629 items from pretrained weights
```

这是预期现象，因为 P2 结构比原 YOLOv8m 多了检测分支，不能全部一一对应加载。不是错误。

## 3. 数据 readiness

最终训练数据：

```text
datasets/stage4_full_tile_sensors3/data.yaml
```

readiness：

```text
reports/stage4_full_tile_sensors3.readiness.json
```

结果：`PASS`

| 检查项 | 当前值 | 门槛 | 结果 |
|---|---:|---:|---|
| train tiny+small fire | 47,119 | 5,000 | PASS |
| train tiny+small smoke | 9,011 | 1,000 | PASS |
| valid tiny+small fire | 27,793 | 300 | PASS |
| valid tiny+small smoke | 3,290 | 100 | PASS |
| total tiny+small other | 15,792 | 1,000 | PASS |
| high-res ratio | 0.5365 | 0.2 | PASS |

第三轮失败的主要数据短板已经被修复：

- 旧数据 `train tiny+small smoke = 118`；
- 当前 Stage4 `train tiny+small smoke = 9,011`；
- 旧数据高分辨率比例为 0；
- 当前 Stage4 高分辨率比例为 0.5365。

## 4. 评估集准备

固定公开 hard holdout：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

特点：

- 500 张；
- 全部来自 FASDD_UAV test；
- 不参与 mixed 训练；
- 100 张空负样本；
- 高分辨率图像 500 张；
- tiny+small fire 977；
- tiny+small smoke 126。

用途：

- 对比第一轮 old YOLOv8m；
- 对比第三轮 YOLOv8m-P2；
- 对比 Stage4 scout；
- 作为是否进入 120 轮长训的主要依据。

## 5. 已采用训练策略

Stage4 当前不是简单混数据，已加入以下训练策略：

1. FASDD_UAV 作为 UAV 火烟主数据；
2. D-Fire 受控混入补 smoke-only 和 none 负样本；
3. full image + tile training；
4. P2 小目标检测头重新验证；
5. smoke-safe augmentation；
6. hard negative 和 empty negative 保留；
7. 每 5 epoch 保存一次；
8. public hard holdout 模型选择。

## 6. 仍需注意的风险

1. 数据量明显增大，scout 也会比第三轮慢。  
   train 从 17,305 增加到 75,744，20 epoch scout 也不是很轻。

2. D-Fire 存在非 UAV 域偏移。  
   目前已限制 D-Fire 数量。如果 scout 出现对地面火更敏感、对 UAV 小火仍不足，需要继续降低 D-Fire 占比。

3. P2 仍不保证一定成功。  
   这次只是给 P2 提供了更合理的数据条件。如果 scout 仍不提升，应使用同一数据再跑普通 YOLOv8m 对照。

4. 当前本机不适合正式训练。  
   本机 PyTorch 是 CPU 版，应在 GPU 服务器上跑。当前本地主要完成数据准备和预检查。

## 7. 训练前最后一条命令

本机或服务器上训练前先运行：

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\19_stage4_preflight_check.ps1
```

Linux：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 bash scripts_linux/19_stage4_preflight_check.sh
```

只有显示 `Ready for Stage4 scout training.` 才启动训练。

## 8. Scout 训练命令

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\12_train_stage4_scout.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3\data.yaml `
  -Name stage4_scout_p2_fasdd_dfire_tile_e20
```

Linux 服务器：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 \
DATA=$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml \
NAME=stage4_scout_p2_fasdd_dfire_tile_e20 \
bash scripts_linux/12_train_stage4_scout.sh
```

## 9. Scout 后评估命令

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\18_eval_public_holdout.ps1 `
  -NewModel <stage4_scout权重路径>
```

必须比较：

- old YOLOv8m 3cls；
- third-round YOLOv8m-P2；
- Stage4 scout epoch 5/10/15/20；
- best.pt；
- last.pt。

## 10. 是否进入 120 轮的判据

只有满足以下条件，才跑 120 轮：

1. public hard holdout 上 small fire/smoke recall 明显超过第一轮；
2. overall recall 不下降；
3. false positives 可控；
4. 视频首次检测时间不晚于旧模型；
5. GPU FPS 可接受。

如果 scout 未达标，不开 120 轮，回到数据比例、tile 策略或普通 YOLOv8m 对照。
