# CenFormer

PyTorch implementation of CenFormer for autism classification from fMRI functional connectomes.

## Dataset

We use the ABIDE dataset with CC200 and AAL116 parcellations. Download the prepared CC200 data from [here](https://drive.google.com/file/d/1rTmBuLbMNu-vW7g43eSu21ur1Sc4oVHh/view?usp=sharing). See [data preparation](data/README.md) for the input format and AAL116 preparation.

Place the dataset files under `data/`:

```text
data/
├── abide.npy       # CC200
└── abide116.npy    # AAL116
```

## Usage

Run the following commands from the project root to train the model:

```bash
# CC200
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --atlas CC200

# AAL116
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --atlas AAL116
```

- **--atlas**: `CC200`, `AAL116`, or `all` (default).
- **--data-dir**: Dataset directory, default `data`.

Each run trains for 50 epochs. Model and training settings are in `source/conf/`.

## Installation

```bash
conda create -n cenformer python=3.9
conda activate cenformer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dependencies

- Python 3.9
- PyTorch 2.5.1
- CUDA 12.1
- NumPy, scikit-learn, Hydra, OmegaConf, W&B

See [requirements.txt](requirements.txt) for package versions.

## Acknowledgement

This implementation builds on Com-BrainTF and [Brain Network Transformer](https://github.com/Wayfear/BrainNetworkTransformer).
