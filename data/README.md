# Data preparation

Run the commands below from the project root.

## CC200 (default)

Download the [prepared CC200 dataset](https://drive.google.com/file/d/1rTmBuLbMNu-vW7g43eSu21ur1Sc4oVHh/view?usp=sharing) and save it as `data/abide.npy`. No additional conversion is needed.

## AAL116

Place these preprocessed arrays in one directory, with matching subject order:

```text
feature_matrix.npy       # FC matrices: (subjects, 116, 116)
time_series_matrix.npy   # Time series: (subjects, time, 116)
labels.npy               # Binary labels: (subjects,)
```

Convert them to the training format:

```bash
python scripts/prepare_aal116.py \
  --input-dir /path/to/processed/abide116 \
  --output data/abide116.npy
```

The script packages existing arrays; it does not preprocess raw fMRI images. Preserve the original subject order, ROI order and label encoding.

Both dataset files store a NumPy dictionary with keys `corr`, `timeseires` (historical spelling), `label`, and `site`.
