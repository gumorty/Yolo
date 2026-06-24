# FireAndSmoke_3 目标主机执行命令

假设你把 `D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3` 打包上传到：

`C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3`

## 1. 进入目录

```powershell
cd C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 2. 检查环境

```powershell
.\scripts\00_check_env.ps1
```

## 3. 从前两轮工程复制数据和权重

```powershell
.\scripts\01_prepare_sources_from_previous.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -FirstRoot "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction\repos" `
  -SecondRoot "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction_Two\repos"
```

## 4. 生成严格数据配置

```powershell
.\scripts\02_build_strict_datasets.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -SelfDataset "C:\Users\jsj506\Desktop\YOLO\Yolo\fire_pic\fire_dataset\fire_dataset_all"
```

## 5. 审计数据集

```powershell
.\scripts\03_audit_datasets.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3"
```

审计报告会输出到：

`C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3\reports`

## 6. 先试跑 5 epoch

```powershell
.\scripts\04_train_stage3_p2_sensors3.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -Epochs 5 `
  -Batch 8 `
  -ImgSize 960
```

如果 3类试跑正常，再试跑 2类：

```powershell
.\scripts\05_train_stage3_p2_sensors2.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -Epochs 5 `
  -Batch 8 `
  -ImgSize 960
```

## 7. 正式过夜训练

```powershell
.\scripts\08_run_stage3_overnight.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -Epochs 120 `
  -Batch 8 `
  -ImgSize 960
```

这会依次训练：

1. `stage3_yolov8m_p2_sensors3_e120`
2. `stage3_yolov8m_p2_sensors2_e120`

如果显存还有余量，可以把 `ImgSize` 提到 `1280`，但建议先完成 `960` 的基线。

## 8. 查看进度

```powershell
$Root="C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3"
Get-Content "$Root\reports\stage3_overnight_status.log" -Tail 20
Get-Content "$Root\reports\stage3_overnight_train.log" -Tail 80
Get-ChildItem "$Root\runs" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,LastWriteTime
```

## 9. 小目标召回评估

训练完成后：

```powershell
.\scripts\06_eval_small_object.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -Model "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3\runs\stage3_yolov8m_p2_sensors3_e120\weights\best.pt" `
  -Data "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3\datasets\strict_sensors_3cls\data.yaml" `
  -ImgSize 960 `
  -Conf 0.20 `
  -Iou 0.50
```

输出报告会包含 tiny/small/medium/large 的召回率。

## 10. RGB-T 数据审计

拿到真实红外热成像数据后，不要直接训练，先审计配对：

```powershell
.\scripts\09_audit_rgbt_pairs.ps1 `
  -Root "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3" `
  -RgbDir "C:\path\to\rgb\images" `
  -ThermalDir "C:\path\to\thermal\images"
```
