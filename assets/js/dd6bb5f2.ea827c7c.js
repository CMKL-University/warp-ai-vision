"use strict";(self.webpackChunkwarp_ai=self.webpackChunkwarp_ai||[]).push([[390],{3905:function(e,t,n){n.d(t,{Zo:function(){return s},kt:function(){return f}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var l=r.createContext({}),u=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},s=function(e){var t=u(e.components);return r.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),d=u(n),f=o,m=d["".concat(l,".").concat(f)]||d[f]||p[f]||a;return n?r.createElement(m,i(i({ref:t},s),{},{components:n})):r.createElement(m,i({ref:t},s))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=d;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:o,i[1]=c;for(var u=2;u<a;u++)i[u]=n[u];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},152:function(e,t,n){n.r(t),n.d(t,{assets:function(){return s},contentTitle:function(){return l},default:function(){return f},frontMatter:function(){return c},metadata:function(){return u},toc:function(){return p}});var r=n(7462),o=n(3366),a=(n(7294),n(3905)),i=["components"],c={},l="PyTorch with CUDA device",u={unversionedId:"lab-book/cuda",id:"lab-book/cuda",title:"PyTorch with CUDA device",description:"Introduction",source:"@site/docs/lab-book/cuda.md",sourceDirName:"lab-book",slug:"/lab-book/cuda",permalink:"/docs/lab-book/cuda",draft:!1,editUrl:"https://github.com/CMKL-University/warp-ai-vision/tree/main/warp-ai/docs/lab-book/cuda.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Translate your site",permalink:"/docs/tutorial-extras/translate-your-site"},next:{title:"Hough Line Transform in OpenCV",permalink:"/docs/lab-book/opencv"}},s={},p=[{value:"Introduction",id:"introduction",level:2},{value:"Getting started with CUDA in PyTorch",id:"getting-started-with-cuda-in-pytorch",level:2}],d={toc:p};function f(e){var t=e.components,n=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"pytorch-with-cuda-device"},"PyTorch with CUDA device"),(0,a.kt)("h2",{id:"introduction"},"Introduction"),(0,a.kt)("p",null,"CUDA is a parallel computing platform and programming model created by NVIDIA. ",(0,a.kt)("a",{parentName:"p",href:"https://developer.nvidia.com/cuda-zone"},"CUDA"),"\n\xa0helps developers speed up their applications by harnessing the power of GPU accelerators."),(0,a.kt)("h2",{id:"getting-started-with-cuda-in-pytorch"},"Getting started with CUDA in PyTorch"),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"torch.cuda")," packages support CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation."),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"To determine if your system supports CUDA, use ",(0,a.kt)("inlineCode",{parentName:"strong"},"is_available()")," function.")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import torch\nprint(f"cuda is available : {torch.cuda.is_available()}")\nprint(f"device name : {torch.cuda.get_device_name(0)}")\n\n# output\n# cuda is available : True\n# device name : NVIDIA A100-SXM4-40GB\n')),(0,a.kt)("p",null,"Now, we can use our CUDA device to handle the model or tensor by using ",(0,a.kt)("inlineCode",{parentName:"p"},"to(device)"),". By default, PyTorch will use our CPU to handle all instances and operations, but with ",(0,a.kt)("inlineCode",{parentName:"p"},"to(device)")," we can transfer all of them to the GPU."),(0,a.kt)("p",null,"By transferring our instances to CUDA devices, they bring in the power of GPU-based parallel processing instead of the usual CPU-based sequential processing in their usual programming workflow."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'# Get cpu or gpu device for training.\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\nprint(f"Using {device} device")\n\n# Creating a test tensor\nx = torch.randint(1, 100, (100, 100))\n\n# Transferring tensor to GPU\nx = x.to(torch.device(\'cuda\'))\n\n# Define model\nclass NeuralNetwork(nn.Module):\n    def __init__(self):\n        super(NeuralNetwork, self).__init__()\n        self.flatten = nn.Flatten()\n        self.linear_relu_stack = nn.Sequential(\n            nn.Linear(28*28, 512),\n            nn.ReLU(),\n            nn.Linear(512, 512),\n            nn.ReLU(),\n            nn.Linear(512, 10)\n        )\n\n    def forward(self, x):\n        x = self.flatten(x)\n        logits = self.linear_relu_stack(x)\n        return logits\n\n# create the model and transfer it to GPU\nmodel = NeuralNetwork().to(device)\nprint(model)\n')),(0,a.kt)("p",null,"This code will create the tensor and neural network model on our GPU. "),(0,a.kt)("blockquote",null,(0,a.kt)("p",{parentName:"blockquote"},"NOTE: Operations between 2 instances must be on the same device.")))}f.isMDXComponent=!0}}]);