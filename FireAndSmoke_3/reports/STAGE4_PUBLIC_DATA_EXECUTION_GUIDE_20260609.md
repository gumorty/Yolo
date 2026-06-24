# Stage4 公开数据执行方案

生成日期：2026-06-09

## 1. 为什么这个流程有用

第三次训练失败的直接证据是：

- 全部训练/验证/测试图像都是 `max_side<=640`，`imgsz=960` 没有真实细节收益；
- `smoke|tiny` 总数只有 5，`smoke|small` 总数只有 170；
- 新 P2 模型 Precision 略升，但 Recall 和小目标召回下降。

所以问题不是训练轮次不够，而是训练数据和评估目标没有对齐。下一轮必须先证明数据先验成立，再短训验证，再长训。

这个流程的作用是：

1. 防止再次用不含小烟/小火的数据训练 P2；
2. 防止公开数据全量混入后造成域偏移；
3. 防止只看 mAP50-95，忽略 small fire/smoke recall；
4. 把论文目标从“模型换结构”落到“数据、训练、评估、预警指标”闭环。

## 2. 当前没有自有 UAV hard frames 怎么办

没有自有 UAV hard frames 时，不能停下来，也不能直接把当前 Sensors valid/test 当论文 hard set。替代方案是：

- 用 FASDD-UAV 作为公开 UAV 火/烟主数据；
- 用 D-Fire 作为火/烟多样性和负样本补充；
- 从公开数据中固定抽出 `public_hard_holdout`，只测试不训练；
- 当前 Sensors 继续作为基础数据和历史对照。

这套 holdout 不能完全替代真实部署数据，但能先解决第三轮暴露的核心问题：训练集和测试集里小火/小烟太少。

## 3. 推荐数据集角色

| 数据集 | 是否进入下一轮主线 | 角色 |
|---|---|---|
| Sensors 3类 | 是 | 历史基线、基础火/烟/other 数据 |
| FASDD-UAV | 是，优先 | 公开 UAV 火/烟目标域数据 |
| D-Fire | 是，受控混入 | 补 fire/smoke 多样性、smoke-only 和 none 负样本 |
| hard negative 自建 | 是 | 类火/类烟干扰物，标成 other |
| VisDrone | 暂不进入主线 | 可选小目标预训练，不直接当火/烟 |
| TinyPerson | 暂不进入主线 | 尺度问题参考，不建议先训练 |
| HIT-UAV / FLAME | 暂不进入 RGB 主线 | 后续热成像风险分支 |

## 4. 下载后目录怎么放

建议统一整理为：

```text
datasets/
  stage4_sources/
    fasdd_uav_curated/
      train/images
      train/labels
      val/images
      val/labels
      data.yaml
    dfire_curated/
      train/images
      train/labels
      val/images
      val/labels
      data.yaml
    hard_negative/
      train/images
      train/labels
      val/images
      val/labels
      data.yaml
```

所有数据统一 YOLO bbox 格式。类别最终必须映射到：

```text
0 fire
1 other
2 smoke
```

D-Fire 和 FASDD-UAV 如果只有两类，则先保持原始 `fire/smoke`，在合并脚本里映射到三类。hard negative 数据如果只是干扰物，可以全部标成 `other`；如果是空负样本，label txt 留空。

## 5. 修改模板

编辑：

```text
configs/stage4_dataset_mix_template.yaml
```

把已经准备好的数据源改为：

```yaml
enabled: true
```

例如 FASDD-UAV：

```yaml
- id: fasdd_uav_curated
  enabled: true
  data: "D:/Researching/Yolo/FireAndSmoke/FireAndSmoke_3/datasets/stage4_sources/fasdd_uav_curated/data.yaml"
  class_map:
    fire: fire
    smoke: smoke
```

D-Fire 可以先限制数量，避免域偏移：

```yaml
max_images:
  train: 5000
  val: 1000
```

## 6. 构建统一三类数据集

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\14_build_stage4_mixed_dataset.ps1 -Overwrite
```

Linux 服务器：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 OVERWRITE=1 bash scripts_linux/14_build_stage4_mixed_dataset.sh
```

输出：

```text
datasets/stage4_mixed_3cls/data.yaml
```

## 7. 建立公开 hard holdout

从 mixed 数据里抽出固定公开 hard set：

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\15_build_public_holdout.ps1 -Overwrite
```

Linux：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 OVERWRITE=1 bash scripts_linux/15_build_public_holdout.sh
```

输出：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

注意：正式训练时，最好从训练源里排除 holdout 中的图像。当前脚本先用于固定评估集构建，后续如果确认数据路径稳定，可以再做严格去重。

## 8. 构建 full+tile 训练集

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\11_build_highres_tiles.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_mixed_3cls\data.yaml `
  -Out D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3 `
  -CopyFull `
  -Overwrite
```

Linux：

```bash
PROJECT=$HOME/gu/projects/FireAndSmoke_3 \
DATA=$PROJECT/datasets/stage4_mixed_3cls/data.yaml \
OUT=$PROJECT/datasets/stage4_full_tile_sensors3 \
COPY_FULL=1 \
OVERWRITE=1 \
bash scripts_linux/11_build_highres_tiles.sh
```

## 9. 数据先验审计

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\13_audit_stage4_dataset.ps1
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\16_check_stage4_readiness.ps1
```

Linux：

```bash
bash scripts_linux/13_audit_stage4_dataset.sh
bash scripts_linux/16_check_stage4_readiness.sh
```

如果 `stage4_readiness.json` 显示 FAIL，不要训练。先补数据。

最低要求建议：

| 检查项 | 建议门槛 |
|---|---:|
| train tiny+small fire | >= 5000 |
| train tiny+small smoke | >= 1000 |
| val tiny+small fire | >= 300 |
| val tiny+small smoke | >= 100 |
| high-res ratio | >= 20% |
| total tiny+small other | >= 1000 |

这些阈值不是论文结果，而是训练前的安全门槛。第三轮没过这些门槛，所以不该期待 P2 明显改善小目标。

## 10. 短训验证

只在 readiness PASS 后运行：

Windows：

```powershell
D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\scripts\12_train_stage4_scout.ps1 `
  -Data D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\datasets\stage4_full_tile_sensors3\data.yaml
```

Linux：

```bash
DATA=$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml bash scripts_linux/12_train_stage4_scout.sh
```

短训只跑 20 轮，每 5 轮保存一次。不要只看 `best.pt`，要比较 epoch 5、10、15、20。

## 11. 进入 120 轮长训的条件

只有满足以下条件，才进入长训：

- public hard holdout 上 small fire/smoke recall 比第一轮明显提升；
- overall recall 不明显下降；
- false positives 可通过连续帧确认压住；
- 视频首次检测时间不晚于第一轮；
- GPU 推理速度可接受。

如果短训没提升，不要加 epoch。继续补 FASDD-UAV、D-Fire、hard negative 和高分辨率小烟样本。

## 12. 吸取论文经验后的最终训练思路

从 SAHI 学到：切片要进入训练，不只是前端推理。

从 CF-YOLO/CFPT 学到：小目标需要跨尺度浅层空间细节和深层语义融合，但结构改动必须建立在小目标样本足够的前提下。

从 SmokeyNet 学到：预警不能只看单帧 mAP，要看 time-to-detection 和连续帧稳定性。

从 Infra-YOLO/FLAME 学到：高温风险不是 RGB 检测能凭空预测的，必须后续引入热成像或辐射热数据，作为独立风险分支。

因此下一次实验主线是：

```text
公开目标域数据整理
-> class×size 审计通过
-> full+tile 训练数据
-> 20轮 scout
-> hard holdout 图片/视频评估
-> 通过后 120轮正式训练
```

这条路线直接针对第三轮失败原因，不再盲目跑长实验。
