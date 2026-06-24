# FireAndSmoke_3 第三轮训练设计

## 目标

训练一个更适合无人机巡检场景的火焰/烟雾检测模型，重点提升：

- 远处小火苗、小烟雾召回；
- 多处火点、多处烟雾同时出现时的完整检测；
- 灯光、反光、车灯、夕阳等 other 干扰抑制；
- 后续接入真实 RGB-Thermal/红外热成像数据的能力。

## 不做什么

本轮不再直接把 Sensors 和自有 fire_pic 全量混合后训练 300 轮。前两轮已经证明，数据规模增加但标注风格不一致时，模型可能变得更保守，甚至损失多目标召回。

本轮也不把人为输入的温度、CO、烟雾浓度等表单数值画成检测框。真实的红外热成像是图像模态，需要成对 RGB/Thermal 数据；普通传感器时序更适合做风险分数，不适合直接生成空间检测框。

## 实验路线

### E1：Sensors 3类 P2 小目标模型

数据：Sensors 3类严格数据集。  
模型：YOLOv8m-P2，类别为 fire / other / smoke。  
初始权重：第一轮 `sensors_yolov8m_3cls_100_best.pt`，如果目标主机没有该权重则退回 `yolov8m.pt`。  
训练尺寸：优先 `imgsz=960`，显存允许时再做 `1280`。

目的：验证增加 P2 检测头是否提升小目标和多目标召回。

### E2：Sensors 2类 P2 对照模型

数据：Sensors binary 2类严格数据集。  
模型：YOLOv8m-P2，类别为 fire / smoke。  
目的：和 E1 比较，判断 other 类对误报抑制和 fire/smoke 召回的影响。

### E3：小目标分桶评估

用 `tools/evaluate_small_objects.py` 对 tiny/small/medium/large 分桶统计召回率。  
这一步比单看 mAP 更贴合你的实际问题，因为你关心的是远处小火苗有没有被找出来。

### E4：切片推理验证

训练完成后在 FastAPI 或独立评估脚本中做：

- full 640；
- full 960；
- full 1280；
- sliced 640 overlap 0.25；
- full 960 + sliced 640。

如果 sliced 明显提升召回，就把它作为系统推理默认的“大场景模式”。

### E5：RGB-T 数据准备

当拿到真实 RGB/Thermal 数据后，先运行 `scripts/09_audit_rgbt_pairs.ps1` 检查：

- RGB 和 Thermal 是否一一配对；
- 文件命名是否一致；
- 分辨率是否一致；
- 是否有漏配；
- 标签是否能对应到两个模态。

只有数据通过审计后，才进入 RGB-T 双流或后期融合训练。

## 成功判据

第三轮模型不只看 mAP：

- Small Object Recall 提升；
- Multi-instance Recall 提升；
- 外部视频中多火点漏检下降；
- 早期烟雾在低阈值 + 连续帧确认下更早报警；
- other 误报不显著增加；
- 推理 FPS 在目标硬件上可接受。
