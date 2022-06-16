# CUDA

CUDA is a parallel computing platform and programming model created by NVIDIA. [CUDA](https://developer.nvidia.com/cuda-zone)
 helps developers speed up their applications by harnessing the power of GPU accelerators.

## **Getting started with CUDA in PyTorch**

`torch.cuda` packages support CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation.

**To determine if your system supports CUDA, use `is_available()` function.**

```python
import torch
print(f"cuda is available : {torch.cuda.is_available()}")
print(f"device name : {torch.cuda.get_device_name(0)}")

# output
# cuda is available : True
# device name : NVIDIA A100-SXM4-40GB
```

Now, we can use our CUDA device to handle the model or tensor by using `to(device)`. By default, PyTorch will use our CPU to handle all instances and operations, but with `to(device)` we can transfer all of them to the GPU.

By transferring our instances to CUDA devices, they bring in the power of GPU-based parallel processing instead of the usual CPU-based sequential processing in their usual programming workflow.

```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Creating a test tensor
x = torch.randint(1, 100, (100, 100))

# Transferring tensor to GPU
x = x.to(torch.device('cuda'))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create the model and transfer it to GPU
model = NeuralNetwork().to(device)
print(model)
```

This code will create the tensor and neural network model on our GPU. 

> NOTE: Operations between 2 instances must be on the same device.
>