# Decomposition of causality by states and interaction type

> **This code now lives in [SURD](https://github.com/ALD-Lab/SURD).** Install it with `pip install surd` and use it with `from surd import states`. The functions and their results are the same as in this repository, which is kept for reference.

_A Python repository for decomposing synergistic, unique, and redundant causalities by states._

## Introduction
We introduce a state-aware causal inference method that quantifies causality in terms of information gain about
future states. The effectiveness of the proposed approach stems from two key features: its ability
to characterize causal influence as a function of the system state, and its capacity to distinguish
both redundant and synergistic effects among variables. The formulation is non-intrusive and requires only pairs of past and future events, facilitating its application in both computational and experimental investigations. The method also identifies the amount of causality that remains unaccounted for due to unobserved variables. The approach can be used to detect causal relationships in systems with multiple variables, dependencies at different time lags, and instantaneous links.

<img width="4499" height="1031" alt="fig_method_states_v1" src="https://github.com/user-attachments/assets/6b04b6e7-78f0-4751-bd65-e6347ae48a6c" />

## System requirements

The method is designed to operate efficiently on standard computing systems. However, the computational demands increase with the complexity of the probability density functions being estimated. To ensure optimal performance, we recommend a minimum of 16 GB of RAM and a quad-core processor with a clock speed of at least 3.3 GHz per core. The performance metrics provided in this repository are based on tests conducted on macOS with an ARM64 architecture and 16 GB of RAM, and on Linux systems running Red Hat version 8.8-0.8. These configurations have demonstrated sufficient performance for the operations utilized by SURD. Users should consider equivalent or superior specifications to achieve similar performance.

## Getting started
### Installation
Install the core library using your preferred package manager:

```bash
# pip (standard)
pip install surd-states

# uv (fast, recommended — see https://github.com/astral-sh/uv)
uv add surd-states

# conda (install pip inside your environment first)
conda create -n surd python=3.11
conda activate surd
pip install surd-states
```

To also install the dependencies needed to run the interactive marimo examples:

```bash
pip install "surd-states[examples]"   # pip
uv add "surd-states[examples]"        # uv
``` 

To install the dependencies needed to build the documentation:

```bash
pip install "surd-states[docs]"       # pip
uv add "surd-states[docs]"            # uv
``` 

Summary:
| Goal | pip | uv | conda |
|------|-----|----|-------|
| Just use the library | `pip install surd-states` | `uv add surd-states` | `conda install surd-states` |
| Run marimo examples | `pip install "surd-states[examples]"` | `uv add "surd-states[examples]"` | `pip install "surd-states[examples]"` inside conda env |
| Build the docs | `pip install "surd-states[docs]"` | `uv add "surd-states[docs]"` | `pip install "surd-states[docs]"` inside conda env |
| Everything | `pip install "surd-states[examples,docs]"` | `uv add "surd-states[examples,docs]"` | same via pip inside conda env |

### Running the examples
The examples/ folder contains three worked examples. You can run them in three ways depending on your preference:

- Option 1 — Jupyter notebooks (familiar interface, no extra install needed) Open any .ipynb file in Jupyter Lab or Jupyter Notebook:

    ```bash
    jupyter lab examples/E01_benchmark_source.ipynb
    ```
- Option 2 — Marimo notebooks (modern reactive interface) First install marimo (included in the examples extras above), then:

    ```bash
    # Open as an interactive notebook in the browser
    marimo edit examples/marimo/E01_benchmark_source.py

    # Or run as a regular Python script
    python examples/marimo/E01_benchmark_source.py
    ```
    With uv you can run directly without a separate install step:

    ```bash
    uv run marimo edit examples/marimo/E01_benchmark_source.py
    ```

- Option 3 — Plain Python script (no notebook tool required) The marimo .py files are valid Python scripts and can be run directly:

    ```bash
    python examples/marimo/E01_benchmark_source.py
    ```
### Building the documentation
First install the docs dependencies, then:

```bash
cd docs
make html        # macOS / Linux
make.bat html    # Windows
``` 

Open ``docs/build/html/index.html`` in your browser to view the result.

## Citation

If you use our method in your research or software, please cite the following paper:

```bibtex
@article{martinez2025,
  author  = {Mart{\'i}nez-S{\'a}nchez, {\'A}lvaro and Lozano-Dur{\'a}n, Adri{\'a}n},
  title   = {Observational causality by states and interaction type for scientific discovery},
  journal = {Communications Physics},
  year    = {2025},
  month   = {Dec},
  day     = {16},
  volume  = {9},
  number  = {1},
  pages   = {15},
  issn    = {2399-3650},
  doi     = {10.1038/s42005-025-02447-w}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
