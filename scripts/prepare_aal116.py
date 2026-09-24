"""Package preprocessed AAL116 arrays in the format used by CenFormer."""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/abide116.npy"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.output.suffix != ".npy":
        parser.error("output must end with .npy")
    fc = np.load(args.input_dir / "feature_matrix.npy")
    ts = np.load(args.input_dir / "time_series_matrix.npy")
    labels = np.load(args.input_dir / "labels.npy").astype(np.int64)
    if fc.ndim != 3 or fc.shape[1:] != (116, 116):
        parser.error("feature_matrix.npy must have shape (subjects, 116, 116)")
    if labels.shape != (len(fc),) or ts.ndim != 3 or len(ts) != len(fc):
        parser.error("time series, connectivity matrices and labels must align by subject")
    if not np.isin(labels, [0, 1]).all():
        parser.error("labels must use 0 and 1")
    # Preserve the original preparation operations and subject order.
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for matrix in fc:
        np.fill_diagonal(matrix, 1.0)
    ts = np.nan_to_num(np.asarray(ts, dtype=np.float32))
    site = np.array([f"c{int(label)}" for label in labels])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, {"timeseires": ts, "corr": fc, "label": labels, "site": site})
    print(f"Saved {len(labels)} subjects to {args.output}")


if __name__ == "__main__":
    main()
