# AK: Attentive Kernels for Information Gathering

> **Great News**: this paper won the [Best Student Paper Award](https://roboticsconference.org/program/awards/) at RSS 2022!

This repository contains the code for reproducing all the figures, animations, and results in the following paper.
```bibtex
@inproceedings{chen2022ak, 
  author = {Weizhe Chen AND Roni Khardon AND Lantao Liu}, 
  title = {{AK: Attentive Kernel for Information Gathering}}, 
  booktitle = {Proceedings of Robotics: Science and Systems}, 
  year = {2022}, 
  address = {New York City, NY, USA}, 
  month = {June}, 
  doi = {10.15607/RSS.2022.XVIII.047} 
} 
```

# Getting Started

This repository depends on [PyPolo](https://github.com/Weizhe-Chen/pypolo) -- an Informative Planning and Uncertainty-Aware Learning Python library -- that does all the heavy lifting. There is no extra requirements after installing PyPolo.

## Create a virtual environment and install PyPolo

We recommend using a virtual environment via `conda` or `mamba`.
I personally prefer `mamba` over `conda` because `mamba` provides the same interface but is much faster.

```bash
# mamba create -n pypolo python=3.8 pip -y
conda create -n pypolo python=3.8 pip -y

# mamba activate pypolo
conda activate pypolo

# Install PyPolo
pip install pypolo
```

## How can I reproduce the results?

This repository follows the following structure.
The `data` folder is shared across different experiments.
To keep minimum dependency, we have provided the preprocessed data.
The pre-processing steps can be found in the commented code, which require some tricky-to-install packages such as `rasterio`.
Each folder serves for one purpose, including benchmarking (`experiments`), ablation study (`ablation`), overfitting analysis (`overfitting`), sensitivity analysis (`sensitivity`), customed step function demo (`step`),  sin function demo (`sin`), and [Mount Saint Helens](https://en.wikipedia.org/wiki/Mount_St._Helens) volcano demo (`volcano`).

```bash
data
├── sin/
├── srtm/
├── step/
└── volcano/
experiments/
├── parse_arguments.py (optional)
├── animate.py (optional)
├── clean.sh
├── configs/
├── main.py
└── reproduce.sh
ablation/
overfitting/
sensitivity/
sin/
step/
volcano/
LICENSE
README.md
```

To reproduce the results in a folder, simply run the corresponding `reproduce.sh` shell script.
The results will be saved to the automatically created `figures` and `outputs` folder in the current directory.

# Videos

<p align="center"><b>Field Experiments</b></p>
<p align="center"><a href="https://www.youtube.com/watch?v=qpoxSF5S9zk"><img src="https://raw.githubusercontent.com/Weizhe-Chen/weizhe-chen.github.io/master/images/heron_quarry.png" alt="drawing" width="400" height="240"></a></p> 

<p align="center"><b>One Example Environment in Simulation</b></p>
<p align="center"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N17E073.png" width="400" height="240"/></p>

Attentive Kernel | RBF Kernel
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/P92J6NmZeK0"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_ak.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/_94lIe7usx8"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_rbf.png" alt="drawing" width="400" height="240"></a>

Gibbs Kernel | Deep Kernel Learning
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/aZ5PXXW-94U"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_gibbs.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/l3lNihEuoQU"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_dkl.png" alt="drawing" width="400" height="240"></a>

# Known Issues
* PyPolo depends on a Python package `bezier` [that does not support pip install on M1 Mac](https://github.com/dhermes/bezier/issues/276).

