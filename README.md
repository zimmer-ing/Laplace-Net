# Breaking Free: Decoupling Forced Systems with Laplace Neural Networks

This repository provides the official implementation, experimental setup, and supplementary material for our ECML PKDD 2025 paper:

**Breaking Free: Decoupling Forced Systems with Laplace Neural Networks**  
Bernd Zimmering, Cecília Coelho, Vaibhav Gupta, Maria Maleshkova, Oliver Niggemann  
Institute for Artificial Intelligence, Helmut Schmidt University Hamburg, Germany  
ECML PKDD 2025, Porto, Portugal  
[Preprint here](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_583.pdf)

## Citation
If you use this code or the datasets in your research, please cite our paper:

```bibtex
@inproceedings{zimmering2025LPNet,
  title={Breaking Free: Decoupling Forced Systems with Laplace Neural Networks},
  author={Zimmering, Bernd and Coelho, Cecília and Gupta, Vaibhav and Maleshkova, Maria and Niggemann, Oliver},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year={2025},
  address={Porto, Portugal},
  publisher={Springer}
}
```
---

Laplace-Net is a modular, solver-free neural architecture for learning forced, delayed, and memory-aware dynamical systems. By leveraging the Laplace transform, it enables efficient training and robust forecasting on complex input signals—decoupling the internal system dynamics from arbitrary excitations and initial states.  
This repository contains all code, configuration files, and datasets needed to reproduce the results from the paper and to apply Laplace-Net to your own dynamical systems.

---

## Repository Structure
The repository is organized as follows:

```
├── configs/               # Configuration files for experiments and model parameters
├── data/                  # Datasets used for training and evaluation
├── results/               # Results of experiments, including trained models and evaluation metrics
├── scripts/               # Executable scripts for running experiments and evaluations
├── Constants.py           # Constants used throughout the project
├── LICENSE                # License file for the project
├── README.md              # This file
├── environment.yml        # Conda environment file for setting up the project dependencies
├── src/                   # Source code for the Laplace Neural Network (Laplace-Net)
│   ├── datasets/          # Base Class and Dataset classes for loading and processing datasets
│   ├── models/            # Base Class and Model classes for implementing Laplace-Net, LNO, and LSTM models
│   ├── experiments/       # Base Class and Experiment classes for training and evaluating models
│   ├── utils/             # Utility functions for data processing, logging, and configuration

```

Here’s a revised “General Code Structure” section, expanded and clarified for an ML researcher. I’ve incorporated details from your repo’s layout, the use of config files, naming conventions via __init__.py registries, and the class hierarchy:

---

## General Code Structure

Laplace-Net is designed for modularity and extensibility, making it straightforward to add new models, datasets, and experimental workflows. The core architecture follows a base-class pattern and uses explicit registries for dynamic selection.

### Base Classes and Hierarchy

- **BaseExperiment**  
  - Orchestrates the end-to-end workflow, including model training and evaluation.
  - Internally utilizes:
    - **BaseModel**: Abstracts common interfaces for training and inference. All model implementations inherit from this, ensuring consistent usage.
    - **BaseDataset**: Handles data loading, preprocessing, and batching. All datasets derive from this class, standardizing data access.

**Hierarchy Overview:**
```
BaseExperiment
  ├── uses BaseModel
  └── uses BaseDataset
```

### Modular Registries and Naming

- **Dynamic Model and Dataset Selection:**  
  - Each module (e.g., `src/models`, `src/datasets`, `src/experiments`) defines a registry in its `__init__.py` file, mapping string names to classes.
  - For example, `DATASET_REGISTRY` and `EXPERIMENTS_REGISTRY` allow code and configs to select implementations by name.
  - This pattern enables easy extension: to add a new model or dataset, simply implement the class and register it.

### Configuration Management

- **Config Files (`configs/` directory):**  
  - All experiments and model parameters are specified via YAML configuration files.
  - The config files use registry names (e.g., `"PredictionPerformance"` for experiments, `"LNO_1D_Duffing_c0"` for datasets) to select the appropriate classes at runtime.
  - This design decouples experiment specification from implementation, simplifying reproducibility and hyperparameter sweeps.

### Summary for ML Researchers

- New models and datasets can be added by subclassing the respective base class and registering in the module’s `__init__.py`.
- Experiments are configured via YAML files, referencing models and datasets by their registry names.
- The base-class and registry system fosters clear separation of concerns and rapid prototyping.

---



## Data

This project uses eight benchmark datasets for the evaluation of Laplace-Net, LNO, and LSTM models.

**Source attribution:**

- **Laplace Neural Operator (LNO) datasets:**  
  The Duffing (c = 0, c = 0.5), Lorenz (ρ = 5, ρ = 10), and driven pendulum (c = 0, c = 0.5) datasets are taken directly from the [Laplace Neural Operator repository](https://github.com/qianyingcao/Laplace-Neural-Operator) by Cao, Goswami, and Karniadakis (MIT License, downloaded February 2025).  
  For details and citation instructions, see [DATASET.md](./data/LNO_Paper/DATASET.md).

- **Spring-Mass-Damper (SMD) and Mackey-Glass datasets:**  
  Both the SMD and Mackey-Glass datasets were generated by the authors specifically for this project using the equations and parameters described in the publication. These datasets are included in this repository for reproducibility.

### Reference

If you use these datasets or this code, please cite:

- **Laplace-Net (this work):**  
  Bernd Zimmering, Cecília Coelho, Vaibhav Gupta, Maria Maleshkova, Oliver Niggemann.  
  "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks."  
  European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), Porto, Portugal, 2025.


- **Laplace Neural Operator (LNO) datasets:**  
  Qianying Cao, Somdatta Goswami, George Em Karniadakis.  
  "Laplace Neural Operator for Solving Differential Equations."  
  [Nature Machine Intelligence 6(6), 631–640 (2024)](https://doi.org/10.1038/s42256-024-00844-4).


---




## Installation

We recommend using **Miniforge** to manage your Python environment for maximum compatibility, especially on MacOS (tested on MacOS 15) and Ubuntu 22.04.

**1. Install Miniforge**  
Follow the [official Miniforge installation instructions](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to download and install Miniforge for your platform.

**2. Create the environment from file**

Clone this repository and set up the environment with:

```bash
conda env create -f environment.yml
conda activate Laplace-Net
```

**3. Manual installation (without conda/miniforge)**

If you do **not** use conda or Miniforge (e.g. you prefer `pip` or another tool), please ensure that you manually install the following packages **with the specified versions** to guarantee reproducibility and compatibility:

| Package            | Version   | pip/conda name      |
|--------------------|-----------|---------------------|
| python             | 3.12      | python              |
| pytorch            | 2.5.1     | torch               |
| plotly             | 5.24.1    | plotly              |
| pandas             | 2.2.3     | pandas              |
| scipy              | 1.14.1    | scipy               |
| optuna             | 4.1.0     | optuna              |
| scikit-learn       | 1.6.1     | scikit-learn        |
| psutil             | 6.1.1     | psutil              |
| pynvml             | 12.0.0    | nvidia-ml-py3 / pynvml|
| tqdm               | 4.67.1    | tqdm                |
| pyyaml             | 6.0.1     | pyyaml              |
| optuna-dashboard   | 0.17.0    | optuna-dashboard    |
| seaborn            | 0.13.2    | seaborn             |

> **Note:**  
> - Use `pip install <package>==<version>` for each package listed above.  
> - Some package names differ between conda and pip (`pyyaml` instead of `yaml`, `torch` for PyTorch).  
> - For **PyTorch**, follow the official [installation instructions](https://pytorch.org/get-started/locally/) to ensure compatibility with your system and GPU drivers (if available).
> - If you encounter issues with dependencies, refer to the provided `environment.yml` as a template for resolving version conflicts.

---

**Tip:**  
For best results and minimal dependency conflicts, we strongly recommend using Miniforge (see above).

---

If you have any questions about manual installation, or need help resolving dependency issues, feel free to [open an issue](#issues) in this repository.

---
## Limitations

- This implementation currently supports only regularly sampled time series data.
- For irregularly sampled data, users must adapt the grid and the routines used for transfer function computation.
- Additionally, the forward Laplace transform is not yet implemented for irregularly sampled data.

## Note on AI Assistance

Some code and documentation in this repository were generated or refined with support from AI tools (e.g., OpenAI GPT-4). Final responsibility for correctness and scientific content rests with the authors.