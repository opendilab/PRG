
# Pretrained Reversible Generation as Unsupervised Visual Representation Learning
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.0-blue)
![Accelerate](https://img.shields.io/badge/accelerate-enabled-yellow?logo=huggingface&logoColor=white)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.01787)
[![demo](https://img.shields.io/badge/PRG-Demo-red)](https://opendilab.github.io/PRG/)

🎉 What I cannot create, I do not understand.


## 🔥 News
-  We first release the version that depends on the [GenerativeRL](https://github.com/opendilab/GenerativeRL.git) library.
- *2025.07.01*: We update PRG camera ready version for ICCV 2025 and release [project website](https://opendilab.github.io/PRG/)
- *2024.11.29*: We release PRG on arxiv.
  


## What is PRG
PRG proposes to turn a pretrained continuous time flow diffusion generator upside down running the model backward produces multi level features that after light fine tuning serve as an unsupervised representation extractor.


## Requirements

- Python 3.10 or later  
- [Conda](https://docs.conda.io/) (optional, but recommended)  
- [GenerativeRL](https://github.com/opendilab/GenerativeRL) (temporarily required)


## Installation

### 1. Prepare the environment
```bash
conda create -n prg-env python=3.10  
conda activate prg-env
```

### 2. Install GenerativeRL
```bash
git clone https://github.com/opendilab/GenerativeRL.git
cd GenerativeRL
pip install -e .
```

### 3. Install PRG
```bash
git clone https://github.com/opendilab/PRG.git
cd PRG
pip install -e .
```

## Training
Simply select the desired configuration file for the corresponding training stage from the [config](https://github.com/opendilab/PRG/tree/main/config) directory, and run your training. For example:

```bash
cd PRG
accelerate launch example.py
```

## Acknowledgement
The Pretrained Reversible Generation codebase is adapted from the following repositories:
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [GenerativeRL](https://github.com/opendilab/GenerativeRL.git)
- [flow_matching](https://facebookresearch.github.io/flow_matching/)
- [SiT](https://github.com/willisma/SiT)
- [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)

A huge thanks to the authors of these projects for their outstanding contributions! 🎉

