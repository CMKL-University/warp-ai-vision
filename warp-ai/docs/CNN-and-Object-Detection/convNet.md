---
sidebar_position: 1
---
# Convolutional Neural Network

## What is Convolutional Neural Network

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

## Convolutional Neural Network using PyTorch

Import the necessary packages for creating a simple neural network.

```python
from torch.autograd import Variable
import torch.nn.functional as F
```

Create our simple convolutional neural network class

```python
class SimpleCNN(torch.nn.Module):
	def __init__(self):
		# define the structure of our network
    super(SimpleCNN, self).__init__()
    #Input channels = 3, output channels = 18
    self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
    self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    #4608 input features, 64 output features (see sizing flow below)
    self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
    #64 input features, 10 output features for our 10 defined classes
    self.fc2 = torch.nn.Linear(64, 10)
	def forward(self, x):
	  x = F.relu(self.conv1(x))
		x = self.pool(x)
	  x = x.view(-1, 18 * 16 *16)
	  x = F.relu(self.fc1(x))
	  #Computes the second fully connected layer (activation applied later)
	  #Size changes from (1, 64) to (1, 10)
	  x = self.fc2(x)
	  return(x)
```

## Feature Visualization on Convolutional Neural Network

Deep Neural Networks are usually treated like “black boxes” due to their **inscrutability** compared to more transparent models, like XGboost or [Explainable Boosted Machines](https://github.com/interpretml/interpret).

However, there is a way to interpret what **each individual filter** is doing in a Convolutional Neural Network, and which kinds of images it is learning to detect by using **[Feature Visualization](https://distill.pub/2017/feature-visualization/)**.

Here are additional resources on convolutional networks and feature visualization.

- [Full articles about convolutional neural network.](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [Feature Visualization](https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af)

**Acknowledgement :** The content of this document has been adapted from these original websites.

- [What is convolutional network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [Convolutional Neural Network using PyTorch](https://www.tutorialspoint.com/pytorch/pytorch_convolutional_neural_network.htm)
- [Feature Visualization on Convolutional Neural Network](https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af)