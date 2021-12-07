# Pipeline for Active Domain Adaptation 

This documents the neccessary steps for running, testing and experimenting with ML models on NERSC. 
It includes the instructions to use OpenMMLab and WandD for unified and consistent model testing and comparison. 
This repo includes several example notebooks for running OpenMMLab tasks. 

## NERSC

### JupyterLab

NERSC computing resources can be easily utilized through (JupyterLab)[https://jupyter.nersc.gov/]. 
File system and command line are all integrated, along with the powerful Jupyter Notebook.

### Conda Environment

It's necessary to use custom conda environment for each machine learning project. Below are the recommended way
to use conda environment on NERSC. Run the following commands in `Terminal`:
```
module load python
conda create --name open-mmlab python=3.8
source activate open-mmlab
```
To deactivate the environment, use
```
(open-mmlab)$ conda deactivate
```
Note: we shouldn't use `conda activate` in NERSC servers. 
Note: it's better practice to move conda setup in `/global/common/software`. However, for our project, we don't have the permission to do so. 
```
conda create --prefix /global/common/software/myproject/myenv python=3.8
source activate /global/common/software/myproject/myenv
```

### PyTorch

PyTorch is an essential framework to build deep learning models. It's also a prerequisite for OpenMMLab.
In our conda environment `open-mmlab`, run the following to install `PyTorch`.
```
(open-mmlab)$ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

### Ipykernel

To register the conda environment with Jupyter Notebook, do the following:
```
(open-mmlab)$ conda install ipykernel
(open-mmlab)$ python -m ipykernel install --user --name open-mmlab --display-name OpenMMLab
```
Then we can refresh the Jupyter Notebook page and select our new conda environment as our kernel. 

## Open MM Lab

OpenMMLab is the open-source library for hundreds of algorithms and models. 
It covers a wide range of research topics of computer vision, 
e.g., classification, detection, segmentation and super-resolution.

### MIM

MIM provides a unified interface for launching and installing OpenMMLab projects and their extensions, 
and managing the OpenMMLab model zoo.

Install MIM in custom conda environment as following:
```
(open-mmlab)$ pip install openmim
```

### MMCV

MMCV is a foundational library for computer vision research and supports many research projects, such as
MMClassification and MMSegmentation.

Install MMCV using MIM in custom conda environment as following:
```
(open-mmlab)$ mim install mmcv-full
(open-mmlab)$ mim list
```

### MMClassification

MMLab's Image Classification Toolbox and Benchmark. It contains various backbones and pretrained models, as well
as training tricks and configs. 

Install MMClassification as following:
```
(open-mmlab)$ mim install mmcls
```

Two example scripts for MMClassification are included in the repo:
- Python API: [mcls-api.ipynb](mcls-api.ipynb)
- Command line: [mcls-shell.ipynb](mcls-shell.ipynb)

## Weights and Bias

The platform for experiment tracking, dataset version, model management and project collaboration. 

<div><img /></div>
<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
<div><img /></div>

Register an account to use the MLOps platform at the [official website](https://wandb.ai/site). In our conda environment, 
```
(open-mmlab)$ pip install wandb --upgrade
```

Add `wandb` monitoring to our code:
```python
# import the library
import wandb

# start a new experiment
wandb.init(project="new-sota-model")

# capture a dictionary of hyperparameters with config
wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

# set up model and data
model, dataloader = get_model(), get_data()

# optional: track gradients
wandb.watch(model)

for batch in dataloader:
  metrics = model.training_step()
  # log metrics inside your training loop to visualize model performance
  wandb.log(metrics)

# optional: save model at the end
model.to_onnx()
wandb.save("model.onnx")
```

For more details, follow this [MLP-FashionMNIST.ipynb](MLP-FashionMNIST.ipynb)