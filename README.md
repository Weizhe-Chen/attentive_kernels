# AK: Attentive Kernels for Information Gathering

> **Great News**: this paper won the [Best Student Paper Award](https://roboticsconference.org/2022/program/awards/) at RSS 2022!

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

1. Creating a Python virtual environment is recommended but not required. I personally prefer `mamba` over `conda` because `mamba` provides the same interface but is much faster.

    ```bash
    conda create -n ak python=3.8
    conda activate ak
    ```

2. Install PyPolo -- a Python library for Robotic Information Gathering.

    > Note: Mac users might need to install the `bezier` library via `BEZIER_NO_EXTENSION=true pip install --upgrade bezier --no-binary=bezier`

    ```bash
    pip install pypolo==0.0.3
    ```

    > Note: The Github repository [PyPolo](https://github.com/Weizhe-Chen/pypolo) has been constantly updated and is NOT compatible with this repository. Please use the PyPolo version specified above. The PyPolo code that corresponds to this paper can be found on https://pypi.org/project/pypolo/

3. Run single-file demo to understand the Attentive Kernel. This demo does not rely on PyPolo.

    ```bash
    cd attentive_kernels/demo
    python main.py
    ```

## Reproducing The Results

This repository follows the following structure.
The `data` folder is shared across different experiments.
To keep minimum dependency, we have provided the preprocessed data.
The pre-processing steps can be found in the commented code, which require some tricky-to-install packages such as `rasterio`.
Each folder serves for one purpose, including benchmarking (`experiments`), ablation study (`ablation`), overfitting analysis (`overfitting`), sensitivity analysis (`sensitivity`), customed step function demo (`step`),  sin function demo (`sin`), and [Mount Saint Helens](https://en.wikipedia.org/wiki/Mount_St._Helens) volcano demo (`volcano`).
Visualization of the weighting and membership functions can be found in the `sin` and `step` folders.

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

You can also run a specific experiment rather than reproducing all the experiments.
For example, the following commands run the Attentive Kernel in environment N17E073 with a myopic informative planner.

```bash
cd attentive_kernels/experiments
python main.py --config ./configs/ak.yaml --env-name N17E073 --strategy myopic --seed 0
# After running the experiment, we can animate the sampling process using the saved data in outputs/ folder.
python animate.py --config ./configs/ak.yaml --env-name N17E073 --strategy myopic --seed 0
```

# Videos

<p align="center"><b>Field Experiments</b></p>
<p align="center"><a href="https://www.youtube.com/watch?v=qpoxSF5S9zk"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/field_experiment_cover.png" alt="drawing" width="400" height="240"></a></p> 

<p align="center"><b>One Example Environment in Simulation</b></p>
<p align="center"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N17E073.png" width="400" height="240"/></p>

Attentive Kernel | RBF Kernel
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/P92J6NmZeK0"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_ak.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/_94lIe7usx8"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_rbf.png" alt="drawing" width="400" height="240"></a>

Gibbs Kernel | Deep Kernel Learning
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/aZ5PXXW-94U"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_gibbs.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/l3lNihEuoQU"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_dkl.png" alt="drawing" width="400" height="240"></a>
