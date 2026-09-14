# DAFRL

Code for **A Unified Data-Aware Fidelity and Regularization Learning Paradigm
for Thick Cloud Removal of Multi-Temporal Remote Sensing Images**.

This repository contains the PyTorch DAFRL/SSFR optimization code and the
MATLAB regression initializer. Training is optimization-based and may require
substantial GPU memory and runtime.

## Setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The input `.mat` file must contain `GT`, `Mask`, and `Ref` arrays. See
[`data/README.md`](data/README.md). Dataset and generated result files are not
tracked by Git.

## Run

```bash
python DAFRL.py --data data/Morocco.mat --output-dir results --epochs 20000
```

Use `--device cpu` or `--device cuda:0` to select the compute device. The
command writes `results/training.log` and `results/prediction.mat`.

## MATLAB initializer

`Regression.m` provides a regression-based initialization function:

```matlab
Xinit = Regression(F, opts);
```

`opts.Omega` is required; `opts.Xtrue` is optional for PSNR reporting.

## Reproducibility and limitations

Results depend on the dataset, device, random initialization, and
hyperparameters. The repository does not include the paper PDF or datasets;
use Git LFS, a release asset, or a data repository if redistribution is
permitted.
