# Running SegRDNet & SoftCAM on APTOS 2019 in Kaggle

## Quick Start (Copy-Paste into Kaggle Notebook)

```python
# 1. Install dependencies
!pip install -q timm pytorch-grad-cam albumentations torchmetrics scikit-learn tensorboard

# 2. Clone both repositories
!git clone https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization.git segrdnet_repo
!git clone https://github.com/chihchiehchen/SoftCAM.git softcam_repo

# 3. Upload the runner scripts to /kaggle/working/ (or copy-paste their content)
# Files needed: run_segrdnet_aptos.py, run_softcam_aptos.py, run_both_aptos.py
```

## Option A: Run Both Models Automatically (Recommended)

```python
# Run comparison (15 epochs each)
%cd /kaggle/working
!python run_both_aptos.py --epochs 15 --batch-size 16 --softcam-network resnet50
```

## Option B: Run Models Individually

### SegRDNet Only
```python
%cd /kaggle/working/segrdnet_repo
!python /kaggle/working/run_segrdnet_aptos.py \
    --data-dir /kaggle/input/competitions/aptos2019-blindness-detection \
    --epochs 15 \
    --batch-size 16 \
    --img-size 512 \
    --output /kaggle/working/segrdnet_output
```

### SoftCAM Only
```python
%cd /kaggle/working/softcam_repo
!python /kaggle/working/run_softcam_aptos.py \
    --data-dir /kaggle/input/competitions/aptos2019-blindness-detection \
    --network resnet50 \
    --epochs 15 \
    --batch-size 16 \
    --img-size 512 \
    --output /kaggle/working/softcam_output
```

## SoftCAM Network Options

Change `--network` to try different backbones:
- `resnet50` (25.6M params) - **Recommended default**
- `resnet101` (44.5M params) - Deeper
- `densenet121` (8.0M params) - Fastest
- `efficientnet_b0` (5.3M params) - Smallest
- `densenet169`, `efficientnet_b3`, `resnet152` - Larger models

## Key Outputs

| Output | Location |
|--------|----------|
| Best model (by QWK) | `{output_dir}/best_model.pth` |
| Training history | `{output_dir}/history.json` |
| Kaggle submission | `{output_dir}/submission.csv` |
| Heatmaps (SoftCAM) | `{output_dir}/heatmaps/` |
| TensorBoard logs | `{output_dir}/` |

## Metrics Computed (Per Epoch)

- **QWK** (Quadratic Weighted Kappa) - Official APTOS metric
- **Accuracy** (Overall + Per-class)
- **AUC** (Macro + Per-class)
- **Confusion Matrix** (5×5)
- **Classification Report** (Precision/Recall/F1 per class)

## GPU Memory Tips

If you hit OOM errors:

```bash
# Reduce batch size
--batch-size 8

# Reduce image size  
--img-size 384

# For SegRDNet: enable gradient accumulation
--grad-accum-steps 2  # Effective batch = 16
```

## Expected Runtime (Kaggle P100/T4)

| Model | Epochs | Batch Size | Est. Time |
|-------|--------|------------|-----------|
| SegRDNet | 15 | 16 | ~25-35 min |
| SoftCAM (ResNet50) | 15 | 16 | ~20-30 min |

## File Structure After Running

```
/kaggle/working/
├── segrdnet_repo/           # Cloned SegRDNet repo
├── softcam_repo/            # Cloned SoftCAM repo
├── run_segrdnet_aptos.py    # SegRDNet runner
├── run_softcam_aptos.py     # SoftCAM runner
├── run_both_aptos.py        # Orchestrator
├── segrdnet_output/         # SegRDNet results
│   ├── segrdnet_aptos_YYYYMMDD-HHMMSS/
│   │   ├── best_model.pth
│   │   ├── history.json
│   │   └── submission.csv
└── softcam_output/          # SoftCAM results
    ├── resnet50/
    │   ├── YYYYMMDD-HHMMSS/
    │   │   ├── best_model.pt
    │   │   ├── submission.csv
    │   │   └── heatmaps/
```

## Troubleshooting

### "Module not found" errors
```python
# Make sure you're in the right directory
%cd /kaggle/working/segrdnet_repo  # for SegRDNet
%cd /kaggle/working/softcam_repo   # for SoftCAM
```

### CUDA out of memory
```bash
--batch-size 8 --img-size 384
```

### Slow data loading
```bash
--workers 2  # Reduce from 4
```

## Submission Format

Both scripts generate `submission.csv` with:
```csv
id_code,diagnosis
ID_abc123,2
ID_def456,0
...
```

Ready for direct Kaggle submission!

---

**Note:** The official APTOS competition metric is **Quadratic Weighted Kappa (QWK)**. Both scripts optimize for this metric.