---
layout: default
title: {{ site.name }}
---

# Abstract

Robotic Information Gathering (RIG) relies on the uncertainty of a probabilistic model to identify critical areas for efficient data collection. Gaussian processes (GPs) with stationary kernels have been widely adopted for spatial modeling. However, real-world spatial data typically does not satisfy the assumption of stationarity, where different locations are assumed to have the same degree of variability. As a result, the prediction uncertainty does not accurately capture prediction error, limiting the success of RIG algorithms. We propose a novel family of nonstationary kernels, named the Attentive Kernel (AK), which is simple, robust, and can extend any existing kernel to a nonstationary one. We evaluate the new kernel in elevation mapping tasks, where AK provides better accuracy and uncertainty quantification over the commonly used RBF kernel and other popular nonstationary kernels. The improved uncertainty quantification guides the downstream RIG planner to collect more valuable data around the high-error area, further increasing prediction accuracy. A field experiment demonstrates that the proposed method can guide an Autonomous Surface Vehicle (ASV) to prioritize data collection in locations with high spatial variations, enabling the model to characterize the salient environmental features.

# Field Experiment
<a href="https://www.youtube.com/embed/qpoxSF5S9zk"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/field_experiment_cover.png" alt="drawing" width="80%"></a>

# Simulated Experiments

## Environment: N17E073

<img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N17E073.png" alt="drawing" width="80%"/>

### Attentive Kernel

<a href="https://www.youtube.com/embed/P92J6NmZeK0"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_ak.png" alt="drawing" width="80%"></a>

### RBF Kernel

<a href="https://www.youtube.com/embed/_94lIe7usx8"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_rbf.png" alt="drawing" width="80%"></a>

### Gibbs Kernel

<a href="https://www.youtube.com/embed/aZ5PXXW-94U"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_gibbs.png" alt="drawing" width="80%"></a>

### Deep Kernel Learning

<a href="https://www.youtube.com/embed/l3lNihEuoQU"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N17E073_dkl.png" alt="drawing" width="80%"></a>

## Environment: N43W080

<img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N43W080.png" alt="drawing" width="80%"/>

### Attentive Kernel

<a href="https://www.youtube.com/embed/_4oyuKxFBkY"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N43W080_ak.png" alt="drawing" width="80%"></a>

### RBF Kernel

<a href="https://www.youtube.com/embed/lx8haGg0aCI"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N43W080_rbf.png" alt="drawing" width="80%"></a>

### Gibbs Kernel

<a href="https://www.youtube.com/embed/5yDTqPvQ9QM"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N43W080_gibbs.png" alt="drawing" width="80%"></a>

### Deep Kernel Learning

<a href="https://www.youtube.com/embed/ZK28mCQYVUQ"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N43W080_dkl.png" alt="drawing" width="80%"></a>


## Environment: N45W123

<img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N45W123.png" alt="drawing" width="80%"/>

### Attentive Kernel

<a href="https://www.youtube.com/embed/BtKDLO1asnk"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N45W123_ak.png" alt="drawing" width="80%"></a>

### RBF Kernel

<a href="https://www.youtube.com/embed/eegdZR_M_zs"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N45W123_rbf.png" alt="drawing" width="80%"></a>

### Gibbs Kernel

<a href="https://www.youtube.com/embed/Z71hU4YbWs0"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N45W123_gibbs.png" alt="drawing" width="80%"></a>

### Deep Kernel Learning

<a href="https://www.youtube.com/embed/Qbk63_b2P1E"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N45W123_dkl.png" alt="drawing" width="80%"></a>


## Environment: N47W124

<img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/envs/N47W124.png" alg="drawing" width="80%"/>

### Attentive Kernel

<a href="https://www.youtube.com/embed/CC9St05e9Ow"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N47W124_ak.png" alt="drawing" width="80%"></a>

### RBF Kernel

<a href="https://www.youtube.com/embed/W1bnETwBtpI"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N47W124_rbf.png" alt="drawing" width="80%"></a>

### Gibbs Kernel

<a href="https://www.youtube.com/embed/mqtxNkIv6bI"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N47W124_gibbs.png" alt="drawing" width="80%"></a>

### Deep Kernel Learning

<a href="https://www.youtube.com/embed/suYUdUDMzTE"><img src="https://raw.githubusercontent.com/Weizhe-Chen/attentive_kernels/gh-pages/assets/play_buttons/N47W124_dkl.png" alt="drawing" width="80%"></a>
