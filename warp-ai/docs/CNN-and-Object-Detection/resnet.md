---
sidebar_position: 3
---

# Deep Residual Network (ResNet)

In this section, we are going to use the pre-trained resnet18 on ImageNet from PyTorch to classify the input image.

Import `torch` and `models` from TorchVision.
```python
import torch
import torchvision.models as models
```

Load the pre-trained resnet18 model from TorchVision.
```python
model = models.resnet18(pretrained=True)
# other resnet model variants from torchvision 
# resnet18 means 18 layers
# models.resnet34(pretrained=True)
# models.resnet50(pretrained=True)
# models.resnet101(pretrained=True)
# models.resnet152(pretrained=True)
print("Model loaded!")
```

Evaluate the pre-trained model.
```python
model.eval()
```

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`  where `H` and `W` are expected to be at least `224`. The images have to be loaded in to a range of `[0, 1]`  and then normalized using `mean =[0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.

---

***Optional:***
We can download an example image from the Pytorch website, but we have already downloaded all the data we need and stored it in the data directory, so we can skip this code section.

```python
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "data/dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

---

Load the `dog.jpg` image from data directory and preprocess the data.
```python
filename = "../data/dog.jpg" # path to data

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename) # stored dog.jpg into input_image
# preprocess the image and transform it to tensor
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
```

Transfer the input and model to our GPU.
```python
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
```

Pass  `input_batch`  into our model.
```python
with torch.no_grad():
    output = model(input_batch)
```

Print the output.
```python
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

There are 1000 image classes in ImageNet. The probabilities of those classes are represented by each element of the output tensor (`probabilities`). We can see how accurate the output is by downloading ImageNet's label and comparing it to our tensor output.

---

***Optional: ***
If you don’t have the ImageNet label on your machine, you can download it from this [link](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt).

- `wget` is a command for retrieving content and files from various web servers.

```python
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

---

Read the categories from ImageNet label file and stores it as list.
```python
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
```

Show the top categories of our input image.
```python
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
		# the output is in 'class' 'probabilities' format 
		# 'probabilites' is floating point number between 0 and 1. 
    print(categories[top5_catid[i]], top5_prob[i].item())
```

## The output
```python
Samoyed 0.8847386837005615
Arctic fox 0.045728735625743866
white wolf 0.04426601156592369
Pomeranian 0.005614196881651878
Great Pyrenees 0.00464773690328002
```

This mean our `dog.jpg` is likely to be Samoyed!

**Acknowledgement**: The content of this chapter has been adapted from the original [PyTorch ResNet](https://pytorch.org/hub/pytorch_vision_resnet).

