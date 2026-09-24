"""Run the original CenFormer training pipeline, one process per seed."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", choices=("CC200", "AAL116", "all"), default="CC200")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides, e.g. preprocess=mixup")
    args = parser.parse_args()
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**32)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds are not allowed")

    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    atlases = ("CC200", "AAL116") if args.atlas == "all" else (args.atlas,)
    filenames = {"CC200": "abide.npy", "AAL116": "abide116.npy"}
    if not args.dry_run:
        for atlas in atlases:
            if not (data_dir / filenames[atlas]).is_file():
                parser.error(f"Missing {data_dir / filenames[atlas]}; see data/README.md")

    env = os.environ.copy()
    env["CENFORMER_DATA_DIR"] = str(data_dir)
    env["WANDB_MODE"] = "offline"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"
    # Keep W&B output with the newly generated experiment artifacts.
    env["WANDB_DIR"] = str(output_dir)
    print(f"Working directory: {root}\nData directory: {data_dir}", flush=True)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for atlas in atlases:
        for seed in args.seeds:
            run_path = output_dir / atlas / f"seed{seed}"
            command = [
                sys.executable, "-B", "-m", "source", f"dataset={atlas}",
                "repeat_time=1", f"seed_offset={seed}",
                # Quote as a Hydra string so spaces and punctuation in paths work.
                f"log_path={json.dumps(str(run_path))}",
                *args.overrides,
            ]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
