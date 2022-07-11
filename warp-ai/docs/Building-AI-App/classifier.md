---
sidebar_position: 2
---
# Image Classification Web App

## Classification Model

We will utilize the ResNet-101 classifier available via TorchVision (more models are available from [PyTorch Hub](https://pytorch.org/hub/)). ResNet-101 is a 101-layer deep learning network proposed in "[Deep Residual Learning for Image Recognition](https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition)". ResNet is one of the most popular architecture for classifiers with over 20,000 citations. The model introduced residual blocks where an identity shortcut connection that skips layer, solving vanishing gradient problem in very deep neural networks.

The following code uses pretrained ResNet-101 model to classify the image into one of the thousand categories available from [ImageNet](https://www.image-net.org/download.php)

```python title="classifier.py"
from torchvision import models, transforms
import torch
from PIL import Image

def predict(image_path):
    # Step 1: Initialize model with the best available weights
    weights = models.ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    img = Image.open(image_path)
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    out = model(batch)
    prob = out.squeeze(0).softmax(0)*100
    _, indices = torch.sort(out, descending=True)
    classes = weights.meta["categories"]
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
```

## Web Application

The web application can be built using basic Python script. The following example include file uploader widget which allows user to upload image to be classfied by the model.

```python title="app.py"
import streamlit as st
from PIL import Image
from classifier import predict

# Initialize app page and file uploader widget
st.title("ResNet Image Classifier")
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    # After a user uploaded the image, open and display the image
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Call the prediction code to classify the image
    st.write("Thinking...")
    labels = predict(file_up)

    # Print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction →", i[0], ",   Score: ", i[1], "%")
```

You can use Streamlit to launch the app. Try uploading a jpg image to see the classfication result.
```bash
streamlit run app.py
```

**Acknowledgement :** The content of this document has been adapted from these original websites.
- [Hasty vision AI Wiki: ResNet](https://wiki.hasty.ai/model-architectures/resnet)
- [Torchvision models and pre-trained weights](https://pytorch.org/vision/main/models.html)
- [Create an Image Classification Web App using PyTorch and Streamlit](https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24)
