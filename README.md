# DoGdiffsea: Frequency‑Aware Diffusion for Underwater Image Enhancement via a Difference of Gaussian

San Zhang, Qianwen Ding, Manli Zhou, Yuhang Ma, Shiliang Zhou

## Abstract
Underwater image enhancement (UIE) remains a fundamental challenge due to complex degradations such as selective light absorption and medium scattering, which lead to severe color casts, loss of contrast, and blurred structural details. Traditional physical-based methods often suffer from limited representational capacity, while recent deep generative models can introduce unwanted artifacts or lack physical interpretability. To bridge this gap, we propose DoGdiffsea, a frequency‑aware diffusion framework grounded in deterministic priors. A Multi‑scale Difference of Gaussian Decomposition (MDOGD) is first employed to losslessly separate the degraded observation into frequency‑ordered components, enabling explicit modeling of illumination drift, structural variations, and high‑frequency residuals. A Deterministic Prior‑Driven Enhancement (DPDE) module then performs frequency‑targeted color correction and contrast compensation. To ensure adaptability across heterogeneous water types, Bayesian Optimization is used to automatically determine optimal enhancement parameters by maximizing expected improvement of fidelity‑related metrics. The refined priors are subsequently injected into a Prior‑Guided Diffusion Model (PGDM) as conditional constraints, reformulating denoising as a physically guided prediction process that restores fine textures while suppressing residual noise. Experiments on UIEB, EUVP, and SUIM‑E confirm highly competitive performance, and ablation studies validate the contributions of frequency‑aware priors and optimization‑driven refinement.

## This codebase was tested with the following environment configurations. It may work with other versions.

- PyTorch  1.11.0
- Python  3.8(ubuntu20.04)
- CUDA  11.3

## Preparing datasets
1. EUVP U60:[data](https://li-chongyi.github.io/proj_benchmark.html)
2. UIEB:[data](https://github.com/JJsnowx/EUVP_Dataset/tree/main/EUVP%20Dataset)
3. SUIM:[data](https://github.com/trentqq/SUIM-E)
4. U45:[data](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-/tree/master/upload/U45)

## Training / Testing
To make use of the [main.py](https://github.com/iAlphaGo/VMD_diff_sea/blob/main/main.py)

## Contact
Should you have any question, please contact <dingqianwen@stu.xupt.edu.cn>







