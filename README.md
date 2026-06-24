# Yolo FireAndSmoke

基于 YOLO 的火焰与烟雾检测项目，针对无人机真实巡检场景中的小目标、多目标、弱烟和红外热成像预警问题。

## 项目结构

```
FireAndSmoke_3/          # 主工程目录
├── configs/             # 训练配置文件
├── scripts/             # PowerShell 执行脚本
├── scripts_linux/       # Linux 执行脚本
├── tools/               # 数据审计、评估、工具脚本
├── deploy_pipeline.py   # 模型部署流水线
├── deploy_pipeline_v2.py# 部署流水线 v2
├── ssh_helper.py        # SSH 辅助工具
├── README.md            # 项目详情
└── VIDEO_EVAL_ANALYSIS.md # 视频评估分析
```

## 主要特性

- 使用 YOLOv8m-P2 结构，增加 P2/4 检测头，提升小目标检测能力
- 2类（fire/other）和 3类（fire/other/smoke）双训练线
- 小目标召回评估（tiny/small/medium/large）
- RGB-Thermal 数据配对审计
- 面向无人机红外热成像融合训练

## 快速开始

详见 [FireAndSmoke_3/README.md](FireAndSmoke_3/README.md)。

## 许可证

MIT
