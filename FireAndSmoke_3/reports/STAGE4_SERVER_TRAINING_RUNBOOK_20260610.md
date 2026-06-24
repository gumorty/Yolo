# Stage4 GPU 服务器训练运行手册

生成日期：2026-06-10

## 1. 服务器硬件判断

根据服务器输出：

| 项目 | 状态 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | 2 × NVIDIA GeForce RTX 4090 |
| GPU memory | GPU0 46GB, GPU1 49GB |
| Driver | 550.144.03 |
| CUDA | 12.4 |
| RAM | 188GiB |
| `/home` free space | 1.4T |
| current GPU processes | none |

硬件结论：可以承担 Stage4 scout 训练。建议使用双卡：

```bash
DEVICE=0,1
BATCH=16
IMGSZ=960
```

## 2. 不能直接开训的前提检查

硬件 OK 不代表立即训练。必须确认以下三项：

1. 项目已上传到：

```bash
/home/uav/gu/projects/FireAndSmoke_3
```

2. 数据集已包含：

```bash
/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml
/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_eval/public_hard_holdout/data.yaml
```

3. 服务器 Python 环境是 CUDA 版 PyTorch，不是 CPU 版。

## 3. 服务器环境检查命令

进入项目：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3
```

检查 Python / PyTorch / Ultralytics：

```bash
python3 - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("cuda version", torch.version.cuda)
    print("gpu count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch error:", repr(e))
try:
    import ultralytics
    print("ultralytics", ultralytics.__version__)
except Exception as e:
    print("ultralytics error:", repr(e))
PY
```

如果 `torch.cuda.is_available()` 不是 `True`，不要训练，先安装 CUDA 版 PyTorch。

## 4. 训练前最终 preflight

```bash
PROJECT=/home/uav/gu/projects/FireAndSmoke_3 bash scripts_linux/19_stage4_preflight_check.sh
```

必须看到：

```text
Ready for Stage4 scout training.
```

否则不要训练。

## 5. 推荐 scout 训练命令

建议先跑 20 轮 scout，不直接 120 轮：

```bash
cd /home/uav/gu/projects/FireAndSmoke_3

PROJECT=/home/uav/gu/projects/FireAndSmoke_3 \
RUN_ROOT=/home/uav/gu/stage4 \
DATA=/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml \
NAME=stage4_scout_p2_fasdd_dfire_tile_e20 \
EPOCHS=20 \
BATCH=16 \
IMGSZ=960 \
DEVICE=0,1 \
WORKERS=8 \
nohup bash scripts_linux/12_train_stage4_scout.sh > /home/uav/gu/stage4/stage4_scout_p2_fasdd_dfire_tile_e20.log 2>&1 &
```

查看日志：

```bash
tail -f /home/uav/gu/stage4/stage4_scout_p2_fasdd_dfire_tile_e20.log
```

查看 GPU：

```bash
nvidia-smi
```

## 6. 如果报文件句柄或 DataLoader 错误

当前服务器 `ulimit -n = 1024`，数据集较大。如果训练中出现 too many open files，先执行：

```bash
ulimit -n 65535
```

再重新启动训练。

## 7. 为什么不直接 120 轮

Stage4 数据先验已经 PASS，但还没有证明训练策略真的优于第一轮和第三轮。必须先通过 public hard holdout：

```text
datasets/stage4_eval/public_hard_holdout/data.yaml
```

scout 通过条件：

- small fire/smoke recall 明显高于第一轮；
- overall recall 不下降；
- false positives 可控；
- 视频首次检测时间不晚于旧模型；
- GPU FPS 可接受。

通过后再跑 120 轮正式训练。

## 8. Scout 完成后的评估

将 `best.pt`、`last.pt`、epoch 5/10/15/20 权重都保留。对 public hard holdout 评估后再决定长训。

如果在服务器上评估，可以使用：

```bash
python tools/compare_detection_configs.py \
  --image-dir datasets/stage4_eval/public_hard_holdout/images \
  --old-model weights/sensors_yolov8m_3cls_100_best.pt \
  --new-model /home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/best.pt \
  --max-images 500 \
  --seed 20260610 \
  --require-small-fire-smoke \
  --out reports/stage4_public_holdout_compare_best.json
```

## 9. 当前判断

硬件层面：可以训练。

项目层面：本地已经完成数据、脚本、readiness、preflight、相对路径修复。

服务器层面：只有在项目上传完整、CUDA PyTorch 可用、服务器 preflight PASS 后，才能启动 scout。
