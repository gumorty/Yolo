# 本地数据集清单与推送说明（2026-07-01）

## 1. 数据推送状态说明

当前 `FireAndSmoke_3/datasets/` 为本地原始数据与中间构建数据目录，已被仓库 `.gitignore` 排除：

```gitignore
FireAndSmoke_3/datasets/
```

截至 2026-07-01，本地数据目录共约 586,279 个文件，合计约 73.18 GB。该体量不适合直接推入 GitHub 普通 Git 历史；若使用 Git LFS，也可能快速超过仓库或账号的 LFS 存储/带宽配额。因此，本次 `7-1` 分支推送保留了数据构建脚本、数据 YAML、manifest、审计报告、训练日志、模型权重、评估结果和论文进展报告，但不直接上传原始图像/标签全集。

## 2. 本地数据目录结构与体量

| 数据目录 | 文件数 | 体量（GB） | 说明 |
|---|---:|---:|---|
| downloads | 118,352 | 17.01 | 外部数据下载与压缩/解压来源 |
| self_fire_pic_2cls_standalone | 1 | 0.00 | 自有二分类数据配置/占位项 |
| stage4_eval | 1,002 | 0.24 | Stage4 public hard holdout 等固定评估数据 |
| stage4_full_tile_sensors3 | 238,632 | 22.52 | Stage4 全图+切片训练数据主版本 |
| stage4_mixed_3cls | 108,422 | 13.70 | Stage4 三分类混合数据 |
| stage4_sources | 101,864 | 17.50 | Stage4 数据源整理目录 |
| stage6_eval | 2,206 | 0.82 | Stage6 固定评估数据 |
| stage6_mining | 8,335 | 0.80 | Stage6 误检挖掘/硬负样本相关数据 |
| stage6_mixed_3cls_union | 2 | 0.00 | Stage6 union 数据配置/占位项 |
| stage6_scout_3cls | 6,630 | 0.49 | Stage6 scout 训练数据 |
| stage6_sources | 831 | 0.11 | Stage6 数据源整理目录 |
| strict_sensors_2cls | 1 | 0.00 | Sensors 二分类严格版本配置/占位项 |
| strict_sensors_3cls | 1 | 0.00 | Sensors 三分类严格版本配置/占位项 |

## 3. 已进入仓库的可复现实验资料

本次分支中保留并推送的材料包括：

- `FireAndSmoke_3/configs/`：数据混合、训练策略、实验矩阵等配置；
- `FireAndSmoke_3/models/`：YOLOv8m-P2 二分类/三分类模型结构 YAML；
- `FireAndSmoke_3/logs/`：Stage6 本地训练与挖掘日志；
- `FireAndSmoke_3/paper_artifacts/`：服务器同步的论文实验材料、评估 JSON、训练结果 CSV、消融实验材料；
- `FireAndSmoke_3/runs_stage6_*`：Stage6 训练结果与权重，`.pt` 文件通过 Git LFS 管理；
- `FireAndSmoke_3/reports/`：数据处理、训练计划、实验分析、论文支撑点和当前状态报告。

## 4. 后续建议

若后续必须把原始数据集也进行远端归档，建议不要直接放入 Git 历史，而采用以下方案之一：

1. 使用 GitHub Release 上传分卷压缩包，并在仓库中维护 SHA256 校验文件；
2. 使用 DVC 或 Git LFS 单独管理数据，提前确认远端存储额度；
3. 将数据保存在服务器/网盘/对象存储中，仓库只保存下载脚本、数据清单、校验值和构建流程。

当前论文复现实验的优先级是保证“数据构建流程、数据配置、审计结果、训练权重、评估结果和实验日志”可追踪，而不是把 73GB 原始数据直接写入 Git 历史。
