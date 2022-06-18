---
sidebar_position: 1
---
# Setup environment for PyTorch

Throughout the following sections, we will use the PyTorch framework for building and developing the tutorials and labs.

To get started on your local machine, please follow this tutorial from PyTorch about [install PyTorch locally.](https://pytorch.org/get-started/locally/) 

### Examples

If your machine is a Linux OS with a [CUDA capable system](https://developer.nvidia.com/cuda-zone), you can use this command to install PyTorch.

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

If your machine is a Windows OS without a [CUDA-capable system](https://developer.nvidia.com/cuda-zone), you can use this command to install PyTorch.

```python
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

For other preferences, follow the PyTorch tutorial to [install PyTorch locally.](https://pytorch.org/get-started/locally/) 

> If you encounter any problems with installing PyTorch, try creating a new Conda environment.
>