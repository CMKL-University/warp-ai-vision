---
sidebar_position: 4
---
# Hough Line Transform in OpenCV

## Introduction to Hough Transform

The Hough Transform is a popular technique to detect any shape, if you can represent that shape in a mathematical form. It can detect the shape even if it is broken or distorted a little bit. We will see how it works for a line using OpenCV. 

```python
import cv2 as cv
import numpy as np
```

We read the sudoku.png that store in our data directory and convert it into gray scale. Next, we use `cv.Canny()` to detect the edge of the image.

```python
img = cv.imread('../data/sudoku.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
```

We use `cv.HoughLines()` function to perform Hough Transform. It returns array of line parametees (rho, theta) values.

- `edges` : binary edge detected image input
- `1` : rho value
- `np.pi/180` : theta
- `200` : threshold

```python
lines = cv.HoughLines(edges,1,np.pi/180,200)
```

Iterate through each (rho, theta) order paired and convert it back to the points on cartesian plane and draw the intersect line between two point on the input image. Finally, we save the output image to our preferred location.

```python
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv.imwrite('../output/houghlines3.jpg',img)
# cv.imshow("Lines",img) 
# cv.waitKey(0)
```

---

## **Probabilistic Hough Transform**

Probabilistic Hough Transform is an optimization of the Hough Transform we saw, it takes only a random subset of points which is sufficient for line detection.

In openCV we can perform the Probabilistic Hough Transform by using `cv.HoughLinesP()` function.

```python
import cv2 as cv
import numpy as np
img = cv.imread('../data/sudoku.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
```

`cv.HoughLinesP()` parameters if similar to `cv.HoughLines()` except it takes two new arguments.

- **minLineLength** - Minimum length of line. Line segments shorter than this are rejected
- **maxLineGap** - Maximum allowed gap between line segments to treat them as a single line

```python
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
```

The `cv.HoughLinesP()` function returns the line's two endpoints, whereas the `cv.HoughLines()` function returns the rho, theta parameters. Then we iterate through each pair of endpoints and draw the line between two endpoints on the input image. Finally, we save our output image as a file.

```python
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv.imwrite('../output/houghlines5.jpg',img)
# cv.imshow("Lines",img) 
# cv.waitKey(0)
```

** Additional Resources :** 
- [Hough Transform Carnegie Mellon University](https://www.cs.cmu.edu/~16385/s17/Slides/5.3_Hough_Transform.pdf)
- [Probabilistic Hough Transform - Iain Macdonald](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/macdonald.pdf)

**Acknowledgement :**  The content of this document has been adapted from the original [Hough Line Transform](https://docs.opencv.org/5.x/d6/d10/tutorial_py_houghlines.html).
