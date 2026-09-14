# DAFRL

Code for **A Unified Data-Aware Fidelity and Regularization Learning Paradigm
for Thick Cloud Removal of Multi-Temporal Remote Sensing Images**.

This repository contains the PyTorch DAFRL optimization code and the
MATLAB regression initializer. Training is optimization-based and may require
substantial GPU memory and runtime.

## Paper

The original paper is available at [IEEE Xplore](https://ieeexplore.ieee.org/document/11299282).

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

## Citation

If you use DAFRL or the related regularization framework, please consider
citing the following papers:

```bibtex
@article{peng2025unified,
  title={A Unified Data-Aware Fidelity and Regularization Learning Paradigm for Thick Cloud Removal of Multitemporal Remote Sensing Images},
  author={Peng, Hao and Huang, Ting-Zhu and Zhao, Xi-Le and Wu, Wei-Hao and Lin, Jie and Ji, Teng-Yu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={63},
  pages={1--13},
  year={2025},
  publisher={IEEE}
}

@article{peng2024deep,
  title={Deep domain fidelity and low-rank tensor ring regularization for thick cloud removal of multitemporal remote sensing images},
  author={Peng, Hao and Huang, Ting-Zhu and Zhao, Xi-Le and Lin, Jie and Wu, Wei-Hao and Li, Li-Yuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```
