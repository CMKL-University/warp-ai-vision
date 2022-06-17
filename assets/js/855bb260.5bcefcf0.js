"use strict";(self.webpackChunkwarp_ai=self.webpackChunkwarp_ai||[]).push([[459],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return f}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var u=r.createContext({}),s=function(e){var t=r.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=s(e.components);return r.createElement(u.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,u=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),d=s(n),f=o,h=d["".concat(u,".").concat(f)]||d[f]||p[f]||a;return n?r.createElement(h,i(i({ref:t},c),{},{components:n})):r.createElement(h,i({ref:t},c))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=d;var l={};for(var u in t)hasOwnProperty.call(t,u)&&(l[u]=t[u]);l.originalType=e,l.mdxType="string"==typeof e?e:o,i[1]=l;for(var s=2;s<a;s++)i[s]=n[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},4376:function(e,t,n){n.r(t),n.d(t,{assets:function(){return c},contentTitle:function(){return u},default:function(){return f},frontMatter:function(){return l},metadata:function(){return s},toc:function(){return p}});var r=n(7462),o=n(3366),a=(n(7294),n(3905)),i=["components"],l={sidebar_position:1},u="Convolutional Neural Network",s={unversionedId:"CNN-and-Object-Detection/convNet",id:"CNN-and-Object-Detection/convNet",title:"Convolutional Neural Network",description:"What is Convolutional Neural Network",source:"@site/docs/CNN-and-Object-Detection/convNet.md",sourceDirName:"CNN-and-Object-Detection",slug:"/CNN-and-Object-Detection/convNet",permalink:"/warp-ai-vision/docs/CNN-and-Object-Detection/convNet",draft:!1,editUrl:"https://github.com/CMKL-University/warp-ai-vision/tree/main/warp-ai/docs/CNN-and-Object-Detection/convNet.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"Classification and Neural Network using PyTorch",permalink:"/warp-ai-vision/docs/Introduction-to-Neural-Network/quickstart"},next:{title:"Deep Residual Network (ResNet)",permalink:"/warp-ai-vision/docs/CNN-and-Object-Detection/resnet"}},c={},p=[{value:"What is Convolutional Neural Network",id:"what-is-convolutional-neural-network",level:2},{value:"Convoluational Neural Network using PyTorch",id:"convoluational-neural-network-using-pytorch",level:2},{value:"Feature Visualization on Convolutional Neural Network",id:"feature-visualization-on-convolutional-neural-network",level:2}],d={toc:p};function f(e){var t=e.components,n=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"convolutional-neural-network"},"Convolutional Neural Network"),(0,a.kt)("h2",{id:"what-is-convolutional-neural-network"},"What is Convolutional Neural Network"),(0,a.kt)("p",null,"A\xa0",(0,a.kt)("strong",{parentName:"p"},"Convolutional Neural Network (ConvNet/CNN)")," is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics."),(0,a.kt)("h2",{id:"convoluational-neural-network-using-pytorch"},"Convoluational Neural Network using PyTorch"),(0,a.kt)("p",null,"Import the necessary packages for creating a simple neural network."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"from torch.autograd import Variable\nimport torch.nn.functional as F\n")),(0,a.kt)("p",null,"Create our simple convolutional neural network class"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"class SimpleCNN(torch.nn.Module):\n    def __init__(self):\n        # define the structure of our network\n    super(SimpleCNN, self).__init__()\n    #Input channels = 3, output channels = 18\n    self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)\n    self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)\n    #4608 input features, 64 output features (see sizing flow below)\n    self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)\n    #64 input features, 10 output features for our 10 defined classes\n    self.fc2 = torch.nn.Linear(64, 10)\n    def forward(self, x):\n      x = F.relu(self.conv1(x))\n        x = self.pool(x)\n      x = x.view(-1, 18 * 16 *16)\n      x = F.relu(self.fc1(x))\n      #Computes the second fully connected layer (activation applied later)\n      #Size changes from (1, 64) to (1, 10)\n      x = self.fc2(x)\n      return(x)\n")),(0,a.kt)("h2",{id:"feature-visualization-on-convolutional-neural-network"},"Feature Visualization on Convolutional Neural Network"),(0,a.kt)("p",null,"Deep Neural Networks are usually treated like \u201cblack boxes\u201d due to their\xa0",(0,a.kt)("strong",{parentName:"p"},"inscrutability"),"\xa0compared to more transparent models, like XGboost or\xa0",(0,a.kt)("a",{parentName:"p",href:"https://github.com/interpretml/interpret"},"Explainable Boosted Machines"),"."),(0,a.kt)("p",null,"However, there is a way to interpret what\xa0",(0,a.kt)("strong",{parentName:"p"},"each individual filter"),"\xa0is doing in a Convolutional Neural Network, and which kinds of images it is learning to detect by using ",(0,a.kt)("strong",{parentName:"p"},(0,a.kt)("a",{parentName:"strong",href:"https://distill.pub/2017/feature-visualization/"},"Feature Visualization")),"."),(0,a.kt)("p",null,"Here are additional resources on convolutional networks and feature visualization."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53"},"Full articles about convolutional neural network.")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af"},"Feature Visualization"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Acknowledgement :")," The content of this document has been adapted from these original websites."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53"},"What is convolutional network")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://www.tutorialspoint.com/pytorch/pytorch_convolutional_neural_network.htm"},"Convolutional Neural Network using PyTorch")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/feature-visualization-on-convolutional-neural-networks-keras-5561a116d1af"},"Feature Visualization on Convolutional Neural Network"))))}f.isMDXComponent=!0}}]);