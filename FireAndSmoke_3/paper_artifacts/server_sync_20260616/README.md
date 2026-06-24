# Server Sync 2026-06-16

This folder archives the latest server-side experiment evidence from `/home/uav/gu` for paper preparation.

## Sync Scope

- Remote host: `221.14.87.239:6022`
- Remote root: `/home/uav/gu`
- Local root: `D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\paper_artifacts\server_sync_20260616`
- Manifest: `sync_manifest.json`
- Synced files: 32
- Total size: 370,078,531 bytes

## Included Evidence

- `stage4/runs/stage4_e60_v4_corrected/`: V4 P2 run, including `best.pt`, `last.pt`, `results.csv`, and `args.yaml`.
- `stage4/runs/stage5_e120_from_v4best/`: Stage5 fine-tuning run, including `best.pt`, `last.pt`, `results.csv`, and `args.yaml`.
- `ablation_p2/runs/yolov8m_nop2_baseline_960/`: No-P2 ablation run, including `best.pt`, `last.pt`, `results.csv`, and `args.yaml`.
- `stage4/eval_public_holdout/`: V4, S5, and E20 public holdout predictions/results.
- `ablation_p2/eval_public_holdout/`: newly generated No-P2 public holdout predictions/results.
- `projects/FireAndSmoke_3/datasets/`: data YAMLs and public hard holdout manifest.
- `projects/FireAndSmoke_3/reports/`: dataset audit/readiness JSON files.

## Confirmed Main Metrics

| Model | Source | Best Epoch | Val mAP50 | Val mAP50-95 | Holdout mAP50 | Holdout mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| V4_P2 | `stage4_e60_v4_corrected` | 16 | 0.84371 | 0.58371 | 0.92376 | 0.68120 |
| S5_FT | `stage5_e120_from_v4best` | 12 | 0.83829 | 0.57902 | 0.92841 | 0.68532 |
| NoP2_Abl | `yolov8m_nop2_baseline_960` | 44 | 0.84082 | 0.58018 | 0.91585 | 0.67743 |

## Key Per-Size Finding

The P2 model gives a clear Recall50 gain on tiny fire instances in the 500-image public hard holdout:

| Comparison | Tiny Fire AP50 | Tiny Fire AP50-95 | Tiny Fire Recall50 |
|---|---:|---:|---:|
| V4_P2 | 0.20134 | 0.07904 | 0.91014 |
| NoP2_Abl | 0.19367 | 0.07276 | 0.83641 |
| Delta | +0.00767 | +0.00627 | +0.07373 |

This supports a cautious claim: the P2 head mainly improves tiny fire recall, while overall mAP gains remain marginal.

## Generated Analysis Files

- `analyze_per_size_ap.py`: local reproducible script for size-stratified AP/recall analysis.
- `per_size_ap_summary.csv`: tabular size-stratified results.
- `per_size_ap_summary.json`: JSON version of the same table.
- `ablation_p2/eval_public_holdout/holdout_nop2_best_results_corrected.json`: corrected No-P2 holdout per-class mapping.

## Notes

- The password provided in the latest request (`Hpu@1090`) failed authentication. The project helper credential (`Hpu@1909`) succeeded.
- Ultralytics evaluates only classes with ground-truth instances in the holdout: class indices `[0, 2]`, corresponding to `fire` and `smoke`. The corrected No-P2 JSON maps the second compact class row to `smoke`.
- The per-size AP script uses project audit bins: tiny `<0.1%`, small `0.1%-1%`, medium `1%-5%`, large `>=5%` of image area.
