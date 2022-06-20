---
sidebar_position: 3
---
# Bitmap, Filters and Image Kernel

## Bitmap

Bitmap, method by which a display space (such as [graphics](https://www.britannica.com/art/graphic-art) image file) is defined, including the [colour](https://www.britannica.com/science/color) of each of its [pixels](https://www.britannica.com/technology/pixel) (or [bits](https://www.britannica.com/technology/bit-communications)). 

Here are additional resources for [bitmaps](http://paulbourke.net/dataformats/bitmaps/).

## Filters

Filtering techniques are use to enhance and modify digital images. Also, images filters are use to blurring and noise reduction, sharpening and edge detection. Image filters are mainly use for suppress high (smoothing techniques) and low frequencies(image enhancement, edge detection). Classification of image filters is as follows.

Here are additional resources for [image processing using spatial filters.](https://towardsai.net/p/computer-vision/image-processing-using-spatial-filters)

### Edge Detection in an Image
The process of image detection involves detecting sharp edges in the image. We’ll be using one such algorithm known as [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector).

```python
import cv2 as cv
import numpy as np

filename = 'cat.jpg'
img = cv.imread(filename)

# Canny edge detection.
edges = cv.Canny(img, 100, 200)

cv.imshow("canny!", edges)
cv.waitKey(0)
```

### Image Blurring
Image Blurring refers to making the image less clear or distinct. It is done with the help of various low pass filter kernels. [For more details about blurring](https://www.geeksforgeeks.org/python-image-blurring-using-opencv/).

```python
import cv2 as cv
import numpy as np

filename = 'cat.jpg'
img = cv.imread(filename)

cv.imshow('Original Image', img)
cv.waitKey(0)
  
# Gaussian Blur
Gaussian = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow('Gaussian Blurring', Gaussian)
cv.waitKey(0)
  
# Median Blur
median = cv.medianBlur(img, 5)
cv.imshow('Median Blurring', median)
cv.waitKey(0)
  
  
# Bilateral Blur
bilateral = cv.bilateralFilter(img, 9, 75, 75)
cv.imshow('Bilateral Blurring', bilateral)
cv.waitKey(0)

cv.destroyAllWindows()
```

## Image Kernel

An image kernel is a small matrix used to apply effects like the ones you might find in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing. They're also used in machine learning for 'feature extraction', a technique for determining the most important portions of an image. In this context the process is referred to more generally as "convolution" (see: [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network).)

Here are additional resources for [image kernel visualization](https://setosa.io/ev/image-kernels/).

### Kernel and Convolutions with OpenCV

Here are some examples of kernels.
```python
# identity kernel
[[0, 0, 0],
 [0, 1, 0],
 [0, 0, 0]]
# Gaussian Blur
[[1,  4,  6,  4, 1],
 [4, 16, 24, 16, 4],
 [6, 24, 36, 24, 6],
 [4, 16, 24, 16, 4],
 [1,  4,  6,  4, 1]] 
```
For more details, visit this [site](https://towardsdatascience.com/basics-of-kernels-and-convolutions-with-opencv-c15311ab8f55).
There are lots of kernels that we can use. Here are [some examples](https://en.wikipedia.org/wiki/Kernel_(image_processing)).

```python
import cv2 as cv
import numpy as np

filename = 'cat.jpg'
img = cv.imread(filename)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.imshow("Image", img)
cv.waitKey(0)
```

**Box Blur Kernel**
```python
# creating box blur kernel
kernel = np.ones((3, 3), np.float32) / 9
print(kernel)

# now apply our box blur kernel with our image using `.filter2D`
result = cv.filter2D(img, -1, kernel)

cv.imshow("box_blur", result)
cv.waitKey(0)
cv.destroyAllWindows()
```

**Sharpen Kernel**
```python
# creating sharpen kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
print(kernel)

# now apply our sharpen kernel with our image using `.filter2D`
result = cv.filter2D(img, -1, kernel)

cv.imshow("sharpen", result)
cv.waitKey(0)
cv.destroyAllWindows()
```


**Acknowledgement :** The content of this document has been adapted from these original website

- [Bitmap](https://www.britannica.com/technology/bitmap)
- [Filters](https://medium.com/@shashikadilhani97/digital-image-processing-filters-832ec6d18a73)
- [Edge Detection](https://www.geeksforgeeks.org/image-processing-in-python-scaling-rotating-shifting-and-edge-detection/?ref=lbp)
- [Filter OpenCV](https://www.geeksforgeeks.org/filter-color-with-opencv/)
- [Image Kernel](https://setosa.io/ev/image-kernels/)
- [Kernel and Convolutions with OpenCV](https://towardsdatascience.com/basics-of-kernels-and-convolutions-with-opencv-c15311ab8f55)