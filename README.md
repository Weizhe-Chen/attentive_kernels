# AK: Attentive Kernels for Information Gathering
This repository contains the code for reproducing the paper `AK: Attentive Kernel for Information Gathering`, published at Robotics: Science and Systems (RSS), 2022.

Field Experiments | Example Environment in Simulation
:-------------------------:|:-------------------------:|
<p align="center"><a href="https://www.youtube.com/embed/XYxEubfIayM"><img src="https://raw.githubusercontent.com/Weizhe-Chen/weizhe-chen.github.io/master/images/heron_quarry.png" alt="drawing" width="400" height="240"></a></p>  | <p align="center"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N17E073.png" width="400" height="240"/></p>

Attentive Kernel | RBF Kernel
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/P92J6NmZeK0"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_ak.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/_94lIe7usx8"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_rbf.png" alt="drawing" width="400" height="240"></a>

Gibbs Kernel | Deep Kernel Learning
:-------------------------:|:-------------------------:|
<br><a href="https://www.youtube.com/embed/aZ5PXXW-94U"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_gibbs.png" alt="drawing" width="400" height="240"></a> | <br><a href="https://www.youtube.com/embed/l3lNihEuoQU"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_dkl.png" alt="drawing" width="400" height="240"></a>

# Should I Install Anything?

This repository depends on [PyPolo](https://github.com/Weizhe-Chen/pypolo) -- an informative planning and learning Python library that does all the heavy lifting. There is no extra requirements after installing PyPolo.

* Python >= 3.6 (To be specific, I was using Python 3.8.13)
* [PyPolo](https://github.com/Weizhe-Chen/pypolo): an informative planning and learning package

# How Can I Reproduce The Results?

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
This repository follows the above structure. The `data` folder is shared across different experiments. To keep minimum dependency, we have provided the preprocessed data. The preprocessing steps can be found in the commented code, which requires some tricky-to-install packages such as `rasterio`. Each folder serves for one experimenting purpose, including benchmarking (experiments), ablation study (ablation), overfitting analysis (overfitting), sensitivity analysis (sensitivity), sin function demo (sin), customed step function demo (step), and [Mount Saint Helens](https://en.wikipedia.org/wiki/Mount_St._Helens) volcano demo (volcano). To reproduce the results in a folder, simply run the corresponding `reproduce.sh` shell script. The results will be saved to the automatically created `figures` and `outputs` folder in the current directory.

