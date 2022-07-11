---
sidebar_position: 1
---
# Building Apps with StreamLit

## Overview

Streamlit provides a quick and easy way for you to create an app for your AI model. The tool can automatically generate a website that incorporate all kinds of inputs and display widgets. You can consult the full documentation on [Streamlit website](https://docs.streamlit.io/library/get-started).

In this section, we will walk through a simple demonstration of an image classification app built with Streamlit and PyTorch.

## Install Streamlit

Make sure you activate your `warp` environment with PyTorch installed. Then import the necessary packages for creating a simple neural network.

```bash
conda activate warp
conda install streamlit
```

Verify that you can successfully launch streamlit app. The following command should open up a web browser pointing to your app.

```bash
streamlit hello
```

## A Simple App
Streamlit can 'magically' display your data on the web page. For example, you can turn the following simple python script into web app.

```python title="demo.py"
import streamlit as st
import torch
import pandas as pd

# Draw a title and some text to the app:
'''
# This is the document title

This is some _markdown_.
'''

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3],'col2': [4,5,6]})
df  # 👈 Draw the dataframe

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
tf = pd.DataFrame(x)
tf # 👈 Draw the tensor as dataframe

x = 10
'x', x  # 👈 Draw the string 'x' and then the value of x

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

fig  # 👈 Draw a Matplotlib chart
```

You can then launch the sample app.
```bash
streamlit run demo.py
```

**Acknowledgement :** The content of this document has been adapted from these original websites.

- [Streamlit Magic](https://docs.streamlit.io/library/api-reference/write-magic/magic)