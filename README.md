# Decomposition of causality by states

_A Python repository for decomposing synergistic, unique, and redundant causalities by states._

## Introduction
We introduce a state-aware causal inference method that quantifies causality in terms of information gain about
future states. The effectiveness of the proposed approach stems from two key features: its ability
to characterize causal influence as a function of the system state, and its capacity to distinguish
both redundant and synergistic effects among variables. The formulation is non-intrusive and requires only pairs of past and future events, facilitating its application in both computational and experimental investigations. The method also identifies the amount of causality that remains unaccounted for due to unobserved variables. The approach can be used to detect causal relationships in systems with multiple variables, dependencies at different time lags, and instantaneous links.

## System requirements

SURD is designed to operate efficiently on standard computing systems. However, the computational demands increase with the complexity of the probability density functions being estimated. To ensure optimal performance, we recommend a minimum of 16 GB of RAM and a quad-core processor with a clock speed of at least 3.3 GHz per core. The performance metrics provided in this repository are based on tests conducted on macOS with an ARM64 architecture and 16 GB of RAM, and on Linux systems running Red Hat version 8.8-0.8. These configurations have demonstrated sufficient performance for the operations utilized by SURD. Users should consider equivalent or superior specifications to achieve similar performance.

## Getting started

After cloning the repository, you can set up the environment needed to run the scripts successfully by following the instructions below. You can create an environment using `conda` with all the required packages by running:
```sh
conda env create -f environment.yml
```
This command creates a new conda environment and installs the packages as specified in the `environment.yml` file in about 50 seconds. After installing the dependencies, make sure to activate the newly created conda environment with:
```sh
conda activate surd
```

## Citation

If you use SURD in your research or software, please cite the following paper:

```bibtex
@article{surd,
author={Mart{\'\i}nez-S{\'a}nchez, {\'A}lvaro and Arranz, Gonzalo and Lozano-Dur{\'a}n, Adri{\'a}n},
title={Decomposing causality into its synergistic, unique, and redundant components},
journal={Nature Communications},
year={2024},
month={Nov},
day={01},
volume={15},
number={1},
pages={9296},
issn={2041-1723},
doi={10.1038/s41467-024-53373-4}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
