# 文献方法到第三轮实验的映射

## 1. Sensors 2024 FireAndSmoke 数据集

论文：A Comparative Performance Evaluation of YOLO-Type Detectors on a New Open Fire and Smoke Dataset  
来源：[MDPI Sensors 2024](https://www.mdpi.com/1424-8220/24/17/5597)，代码/展示：[NEWFireSmokeDataset_YoloModels](https://github.com/CostiCatargiu/NEWFireSmokeDataset_YoloModels)，[AllYoloModels](https://github.com/CostiCatargiu/AllYoloModels)

该论文的价值不只是 YOLOv8/YOLOv10 分数，而是数据集设计：fire、smoke、other 三类覆盖多场景、多距离、昼夜、室内外、无人机和远景，并把夕阳、车灯、路灯、反光、电子屏等容易误报的目标标为 other。论文还说明数据来自 1200 多个视频和公开网络资源，包含 2 万多张图和 9 万多个实例，并比较了 YOLOv5 到 YOLOv10 等模型。

对本项目的启发：

- 继续保留 3类实验，other 是误报控制的核心。
- 不要把没有 other 的自有数据粗暴混入 3类数据。
- 评估不能只看整体 mAP，要看 day/night、small object、多目标和误报。

## 2. SAHI 小目标切片推理

论文：Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection  
来源：[arXiv 2202.06934](https://arxiv.org/abs/2202.06934)

SAHI 的核心思想是把高分辨率大图切成重叠小块分别检测，再合并检测框。论文指出远处小目标在原图中像素少、细节不足，是常规检测器的困难点；切片推理和切片微调可以显著提升小目标 AP。

对本项目的启发：

- 大场景无人机视频不能只用 full 640。
- 第三轮训练之外，必须保留 full 960/1280 和 sliced 640/768 的推理评估。
- 如果切片推理明显提升多火点召回，说明问题主要来自输入分辨率，而不是必须立刻换模型。

## 3. RGB-Thermal / UAV 热红外数据

代表数据与论文：

- FLAME：无人机火灾航拍数据，包含可见光和热红外/热图方向的数据基础。来源：[FLAME dataset page](https://experts.nau.edu/en/datasets/the-flame-dataset-aerial-imagery-pile-burn-detection-using-drones-2/)，[ScienceDirect Data in Brief](https://www.sciencedirect.com/science/article/pii/S1389128621001201)
- FireMan-UAV-RGBT：无人机 RGB + Thermal 视频数据集，并评估 ResNet50 和 YOLOv8 的单模态/多模态检测。来源：[CoLab/DOI page](https://colab.ws/articles/10.1109%2Fetfa61755.2024.10710657)，[Zenodo dataset](https://zenodo.org/records/13732947)
- UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion Model：构建多场景 RGB-Thermal 森林火灾航拍数据，并基于 YOLOv11 做可见光、热红外和融合实验。来源：[Remote Sensing 2025](https://www.mdpi.com/2072-4292/17/15/2593)

对本项目的启发：

- 红外热成像不能用人为输入数值代替；必须拿到真实热红外图像/视频或配对 RGB-T 数据。
- 近期主流方向是 RGB 和 Thermal 的特征级/中期融合，或者先做单模态 RGB、单模态 TIR，再做后期融合对照。
- 第三轮工程先准备 RGB-T 配对审计脚本，确认图像对齐、命名、尺寸和标签后再训练双流模型。

## 4. 改进 YOLO 的常见有效方向

代表工作：

- FireYOLO-Lite：使用轻量化骨干、注意力机制和小目标相关损失改进森林火灾检测。来源：[Forests 2024](https://www.mdpi.com/1999-4907/15/7/1244)
- Improved YOLOv8 wildfire smoke detection：面向 UAV 图像的烟雾检测，使用 BiFormer、Ghost Shuffle 等结构增强。来源：[PubMed](https://pubmed.ncbi.nlm.nih.gov/37896467/)
- GCM-YOLO / YOLOv8-CBAM / LUFFD-YOLO 等工作普遍围绕注意力、多尺度融合、轻量化和小目标结构改进。来源：[Applied Sciences 2024 GCM-YOLO](https://www.mdpi.com/2076-3417/14/16/6878)，[LUFFD-YOLO DOAJ](https://doaj.org/article/d12477b51bd44fbab9fad20c2b2a9f92)

第三轮选择：

优先实现 P2 小目标检测头和严格数据评估，而不是一开始就改 loss/attention。原因是 P2 改动清晰、可复现、对无人机远景小目标直接相关；attention 和 NWD 属于第四轮或消融扩展，适合在第三轮确认小目标瓶颈后再加入。
