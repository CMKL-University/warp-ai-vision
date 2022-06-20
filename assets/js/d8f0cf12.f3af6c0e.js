"use strict";(self.webpackChunkwarp_ai=self.webpackChunkwarp_ai||[]).push([[405],{3905:function(e,n,t){t.d(n,{Zo:function(){return u},kt:function(){return h}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function p(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var l=r.createContext({}),s=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},u=function(e){var n=s(e.components);return r.createElement(l.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,u=p(e,["components","mdxType","originalType","parentName"]),m=s(t),h=o,g=m["".concat(l,".").concat(h)]||m[h]||c[h]||a;return t?r.createElement(g,i(i({ref:n},u),{},{components:t})):r.createElement(g,i({ref:n},u))}));function h(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=t.length,i=new Array(a);i[0]=m;var p={};for(var l in n)hasOwnProperty.call(n,l)&&(p[l]=n[l]);p.originalType=e,p.mdxType="string"==typeof e?e:o,i[1]=p;for(var s=2;s<a;s++)i[s]=t[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},1444:function(e,n,t){t.r(n),t.d(n,{assets:function(){return u},contentTitle:function(){return l},default:function(){return h},frontMatter:function(){return p},metadata:function(){return s},toc:function(){return c}});var r=t(7462),o=t(3366),a=(t(7294),t(3905)),i=["components"],p={sidebar_position:4},l="Hough Line Transform in OpenCV",s={unversionedId:"Basic-Computer-Vision/opencv",id:"Basic-Computer-Vision/opencv",title:"Hough Line Transform in OpenCV",description:"Introduction to Hough Transform",source:"@site/docs/Basic-Computer-Vision/opencv.md",sourceDirName:"Basic-Computer-Vision",slug:"/Basic-Computer-Vision/opencv",permalink:"/warp-ai-vision/docs/Basic-Computer-Vision/opencv",draft:!1,editUrl:"https://github.com/CMKL-University/warp-ai-vision/tree/main/warp-ai/docs/Basic-Computer-Vision/opencv.md",tags:[],version:"current",sidebarPosition:4,frontMatter:{sidebar_position:4},sidebar:"tutorialSidebar",previous:{title:"Bitmap, Filters and Image Kernel",permalink:"/warp-ai-vision/docs/Basic-Computer-Vision/bitmap-filter-image-kernel"},next:{title:"Setup environment for PyTorch",permalink:"/warp-ai-vision/docs/Introduction-to-Neural-Network/setup-pytorch"}},u={},c=[{value:"Introduction to Hough Transform",id:"introduction-to-hough-transform",level:2},{value:"<strong>Probabilistic Hough Transform</strong>",id:"probabilistic-hough-transform",level:2}],m={toc:c};function h(e){var n=e.components,t=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},m,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"hough-line-transform-in-opencv"},"Hough Line Transform in OpenCV"),(0,a.kt)("h2",{id:"introduction-to-hough-transform"},"Introduction to Hough Transform"),(0,a.kt)("p",null,"The Hough Transform is a popular technique to detect any shape, if you can represent that shape in a mathematical form. It can detect the shape even if it is broken or distorted a little bit. We will see how it works for a line using OpenCV. "),(0,a.kt)("p",null,"To use OpenCV in Python, you need to install the openCV module using ",(0,a.kt)("inlineCode",{parentName:"p"},"conda")," command."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"conda install opencv\n")),(0,a.kt)("p",null,"Now you can import OpenCV into your Python code."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import cv2 as cv\nimport numpy as np\n")),(0,a.kt)("p",null,"We read the sudoku.png that store in your directory and convert it into gray scale. Next, we use ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.Canny()")," to detect the edge of the image."),(0,a.kt)("blockquote",null,(0,a.kt)("p",{parentName:"blockquote"},"For sudoku.png, you can choose any from ",(0,a.kt)("a",{parentName:"p",href:"https://www.google.com/search?q=sudoku+opencv&tbm=isch&ved=2ahUKEwiFncf3wrv4AhX6hNgFHcBeD6kQ2-cCegQIABAA&oq=sudoku+opencv&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgYIABAeEAg6CAgAEIAEELEDOgQIABAeOgYIABAeEAU6BAgAEBhQ4QFYvRVg4xZoAHAAeACAAXKIAYIFkgEDNy4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=2COwYoXRDPqJ4t4PwL29yAo&bih=699&biw=1440"},"Google"),". ")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"img = cv.imread('path/to/sudoku.png')\ngray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)\nedges = cv.Canny(gray,50,150,apertureSize = 3)\n")),(0,a.kt)("p",null,"We use ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLines()")," function to perform Hough Transform. It returns array of line parameters (rho, theta) values."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"edges")," : binary edge detected image input"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"1")," : rho value"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"np.pi/180")," : theta"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"200")," : threshold")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"lines = cv.HoughLines(edges,1,np.pi/180,200)\n")),(0,a.kt)("p",null,"Iterate through each (rho, theta) order paired and convert it back to the points on cartesian plane and draw the intersect line between two point on the input image. Finally, we save the output image to our preferred location."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"for line in lines:\n    rho,theta = line[0]\n    a = np.cos(theta)\n    b = np.sin(theta)\n    x0 = a*rho\n    y0 = b*rho\n    x1 = int(x0 + 1000*(-b))\n    y1 = int(y0 + 1000*(a))\n    x2 = int(x0 - 1000*(-b))\n    y2 = int(y0 - 1000*(a))\n    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)\ncv.imwrite('houghlines3.jpg',img)\n# cv.imshow(\"Lines\",img) \n# cv.waitKey(0)\n")),(0,a.kt)("hr",null),(0,a.kt)("h2",{id:"probabilistic-hough-transform"},(0,a.kt)("strong",{parentName:"h2"},"Probabilistic Hough Transform")),(0,a.kt)("p",null,"Probabilistic Hough Transform is an optimization of the Hough Transform we saw, it takes only a random subset of points which is sufficient for line detection."),(0,a.kt)("p",null,"In openCV we can perform the Probabilistic Hough Transform by using ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLinesP()")," function."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import cv2 as cv\nimport numpy as np\nimg = cv.imread('path/to/sudoku.png')\ngray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)\nedges = cv.Canny(gray,50,150,apertureSize = 3)\n")),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLinesP()")," parameters if similar to ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLines()")," except it takes two new arguments."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"minLineLength"),"\xa0- Minimum length of line. Line segments shorter than this are rejected"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"maxLineGap")," - Maximum allowed gap between line segments to treat them as a single line")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)\n")),(0,a.kt)("p",null,"The ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLinesP()")," function returns the line's two endpoints, whereas the ",(0,a.kt)("inlineCode",{parentName:"p"},"cv.HoughLines()")," function returns the rho, theta parameters. Then we iterate through each pair of endpoints and draw the line between two endpoints on the input image. Finally, we save our output image as a file."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"for line in lines:\n    x1,y1,x2,y2 = line[0]\n    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)\ncv.imwrite('houghlines5.jpg',img)\n# cv.imshow(\"Lines\",img) \n# cv.waitKey(0)\n")),(0,a.kt)("p",null,"Here are the tutorial for lane detection, one of the Hough Transform applications."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://machinelearningknowledge.ai/lane-detection-tutorial-in-opencv-python-using-hough-transform/"},"Lane Detection Tutorial in OpenCV Python using Hough Transform"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Additional Resources :")," "),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://www.cs.cmu.edu/~16385/s17/Slides/5.3_Hough_Transform.pdf"},"Hough Transform Carnegie Mellon University")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/macdonald.pdf"},"Probabilistic Hough Transform - Iain Macdonald"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Acknowledgement :"),"  The content of this document has been adapted from the original ",(0,a.kt)("a",{parentName:"p",href:"https://docs.opencv.org/5.x/d6/d10/tutorial_py_houghlines.html"},"Hough Line Transform"),"."))}h.isMDXComponent=!0}}]);