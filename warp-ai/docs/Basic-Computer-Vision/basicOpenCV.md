---
sidebar_position: 1
---
# Operations with Images in OpenCV

To use OpenCV in Python, you need to install the openCV module using `conda` command.

```bash
conda install opencv
```

Now you can import OpenCV into your Python code.

```python
import cv2 as cv
```

## Input/Output

To load an image from a file:

```python
# Load 'test.jpg' image from your current directory
img = cv.imread("test.jpg")
# Load 'test.jpg' image from 'pictures' directory
img = cv.imread("pictures/test.jpg") 

# Load image from 'filename' variable
filename = "test.jpg"
img = cv.imread(filename)
```

If you read a jpg file, a 3 channel image is created by default. If you need a grayscale image, use:

```python
filename = "test.jpg"
# read test.jpg but it read our image as grayscale
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

# print shape of image
print(img.shape)
```

To save/write your image from OpenCV:

```python
# write 'img' to your path
cv.imwrite('test2.jpg', img)
```

To show your image:

```python
# show 'img' with title name 'test show image'
cv.imshow('test show image',img)
# show image for 5 seconds (5000 ms) before it automatically close it
cv.waitKey(5000)
# or you can use this line, it will wait until you pressed any key
cv.waitKey(0)

cv.destroyAllWindows()
```

## Color Spaces in OpenCV

**Color spaces** are a way to represent the color channels present in the image that gives the image that particular hue. **BGR color space:** OpenCV’s default color space is RGB. However, it actually stores color in the BGR format. 

To visualize the different color channels of RGB image

```python
import cv2 as cv

# you can use any image on your computer
img = cv.imread('cat.jpg')
# seperate each channel of RGB image
B, G, R = cv.split(img)

# show each channel
cv.imshow("original", img)
cv.waitKey(0)
 
cv.imshow("blue", B)
cv.waitKey(0)
 
cv.imshow("Green", G)
cv.waitKey(0)
 
cv.imshow("red", R)
cv.waitKey(0)
 
cv.destroyAllWindows()
```

## Arithmetic Operations on Images

We can applied arithmetic operations like Addition, Subtraction, and Bitwise Operation to our input images.

### Addition

```python
cv.add(img1, img2)
```

### Subtraction

```python
cv.subtract(img1, img2)
```

### Bitwise
> Note: In bitwise operation, both images must have the same dimensions.

Parameters for AND, OR, XOR 
`source1` : First Input Image array(Single-channel, 8-bit or floating-point) 
`source2` : Second Input Image array(Single-channel, 8-bit or floating-point) 
`dest` : Output array (Similar to the dimensions and type of Input image array) 
`mask`: Operation mask, Input / output 8-bit single-channel mask 
```python
dest_and =  cv.bitwise_and(img1, img2, mask=None)
dest_or = cv.bitwise_or(img2, img1, mask = None)
dest_xor = cv.bitwise_xor(img1, img2, mask = None)
```
Parameters for NOT
`source` : Input Image array(Single-channel, 8-bit or floating-point) 
`dest` : Output array (Similar to the dimensions and type of Input image array) 
`mask` : Operation mask, Input / output 8-bit single-channel mask 
```python
dest_not1 = cv.bitwise_not(img1, mask = None)
dest_not2 = cv.bitwise_not(img2, mask = None)
```

##Image Processing in Python

### Scaling an Image

```python
import cv2 as cv
import numpy as np

filename = 'cat.jpg'
img = cv.imread(filename)

# Get number of pixel horizontally and vertically
(height, width) = img.shape[:2]

res = cv.resize(img, (int(width / 2), int(height / 2)), interpolation = cv.INTER_CUBIC)

cv.imwrite("result.jpg", res)
```

Now you can check the dimensions of original and result image.

## Rotating an Image

```python
import cv2 as cv
import numpy as np

filename = 'cat.jpg'
img = cv.imread(filename)

# Shape of image in terms of pixels.
(rows, cols) = img.shape[:2]

# getRotationMatrix2D creates a matrix needed for transformation.
# We want matrix for rotation w.r.t center to 45 degree without scaling.
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
res = cv.warpAffine(img, M, (cols, rows))

cv.imshow("rotated!", res)
cv.waitKey(0)

```

**Acknowledgement :** The content of this document has been adapted from the original [OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/) from GeeksforGeeks