"use strict";(self.webpackChunkwarp_ai=self.webpackChunkwarp_ai||[]).push([[4],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return d}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),l=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=l(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,c=p(e,["components","mdxType","originalType","parentName"]),m=l(n),d=a,h=m["".concat(s,".").concat(d)]||m[d]||u[d]||o;return n?r.createElement(h,i(i({ref:t},c),{},{components:n})):r.createElement(h,i({ref:t},c))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=m;var p={};for(var s in t)hasOwnProperty.call(t,s)&&(p[s]=t[s]);p.originalType=e,p.mdxType="string"==typeof e?e:a,i[1]=p;for(var l=2;l<o;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},2213:function(e,t,n){n.r(t),n.d(t,{assets:function(){return c},contentTitle:function(){return s},default:function(){return d},frontMatter:function(){return p},metadata:function(){return l},toc:function(){return u}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),i=["components"],p={sidebar_position:3},s="Deep Residual Network (ResNet)",l={unversionedId:"CNN-and-Object-Detection/resnet",id:"CNN-and-Object-Detection/resnet",title:"Deep Residual Network (ResNet)",description:"In this section, we are going to use the pre-trained resnet18 on ImageNet from PyTorch to classify the input image.",source:"@site/docs/CNN-and-Object-Detection/resnet.md",sourceDirName:"CNN-and-Object-Detection",slug:"/CNN-and-Object-Detection/resnet",permalink:"/warp-ai-vision/docs/CNN-and-Object-Detection/resnet",draft:!1,editUrl:"https://github.com/CMKL-University/warp-ai-vision/tree/main/warp-ai/docs/CNN-and-Object-Detection/resnet.md",tags:[],version:"current",sidebarPosition:3,frontMatter:{sidebar_position:3},sidebar:"tutorialSidebar",previous:{title:"Convolutional Neural Network",permalink:"/warp-ai-vision/docs/CNN-and-Object-Detection/convNet"},next:{title:"Tutorial - Basics",permalink:"/warp-ai-vision/docs/category/tutorial---basics"}},c={},u=[{value:"The output",id:"the-output",level:2}],m={toc:u};function d(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"deep-residual-network-resnet"},"Deep Residual Network (ResNet)"),(0,o.kt)("p",null,"In this section, we are going to use the pre-trained resnet18 on ImageNet from PyTorch to classify the input image."),(0,o.kt)("p",null,"Import ",(0,o.kt)("inlineCode",{parentName:"p"},"torch")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"models")," from TorchVision."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import torch\nimport torchvision.models as models\n")),(0,o.kt)("p",null,"Load the pre-trained resnet18 model from TorchVision."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'model = models.resnet18(pretrained=True)\n# other resnet model variants from torchvision \n# resnet18 means 18 layers\n# models.resnet34(pretrained=True)\n# models.resnet50(pretrained=True)\n# models.resnet101(pretrained=True)\n# models.resnet152(pretrained=True)\nprint("Model loaded!")\n')),(0,o.kt)("p",null,"Evaluate the pre-trained model."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"model.eval()\n")),(0,o.kt)("p",null,"All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"(3 x H x W)"),"  where\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"H"),"\xa0and\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"W"),"\xa0are expected to be at least\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"224"),". The images have to be loaded in to a range of\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"[0, 1]"),"  and then normalized using\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"mean =[0.485, 0.456, 0.406]"),"\xa0and\xa0",(0,o.kt)("inlineCode",{parentName:"p"},"std = [0.229, 0.224, 0.225]"),"."),(0,o.kt)("hr",null),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("em",{parentName:"strong"},"Optional:")),"\nWe can download an example image from the Pytorch website, but we have already downloaded all the data we need and stored it in the data directory, so we can skip this code section."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'import urllib\nurl, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "data/dog.jpg")\ntry: urllib.URLopener().retrieve(url, filename)\nexcept: urllib.request.urlretrieve(url, filename)\n')),(0,o.kt)("hr",null),(0,o.kt)("p",null,"Load the ",(0,o.kt)("inlineCode",{parentName:"p"},"dog.jpg")," image from data directory and preprocess the data."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'filename = "../data/dog.jpg" # path to data\n\n# sample execution (requires torchvision)\nfrom PIL import Image\nfrom torchvision import transforms\ninput_image = Image.open(filename) # stored dog.jpg into input_image\n# preprocess the image and transform it to tensor\npreprocess = transforms.Compose([\n    transforms.Resize(256),\n    transforms.CenterCrop(224),\n    transforms.ToTensor(),\n    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n])\ninput_tensor = preprocess(input_image)\ninput_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model\n')),(0,o.kt)("p",null,"Transfer the input and model to our GPU."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"if torch.cuda.is_available():\n    input_batch = input_batch.to('cuda')\n    model.to('cuda')\n")),(0,o.kt)("p",null,"Pass  ",(0,o.kt)("inlineCode",{parentName:"p"},"input_batch"),"  into our model."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"with torch.no_grad():\n    output = model(input_batch)\n")),(0,o.kt)("p",null,"Print the output."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes\nprint(output[0])\n# The output has unnormalized scores. To get probabilities, you can run a softmax on it.\nprobabilities = torch.nn.functional.softmax(output[0], dim=0)\nprint(probabilities)\n")),(0,o.kt)("p",null,"There are 1000 image classes in ImageNet. The probabilities of those classes are represented by each element of the output tensor (",(0,o.kt)("inlineCode",{parentName:"p"},"probabilities"),"). We can see how accurate the output is by downloading ImageNet's label and comparing it to our tensor output."),(0,o.kt)("hr",null),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("em",{parentName:"strong"},"Optional: ")),"\nIf you don\u2019t have the ImageNet label on your machine, you can download it from this ",(0,o.kt)("a",{parentName:"p",href:"https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"},"link"),"."),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"wget")," is a command for retrieving content and files from various web servers.")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt\n")),(0,o.kt)("hr",null),(0,o.kt)("p",null,"Read the categories from ImageNet label file and stores it as list."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'# Read the categories\nwith open("imagenet_classes.txt", "r") as f:\n    categories = [s.strip() for s in f.readlines()]\n')),(0,o.kt)("p",null,"Show the top categories of our input image."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# Show top categories per image\ntop5_prob, top5_catid = torch.topk(probabilities, 5)\nfor i in range(top5_prob.size(0)):\n        # the output is in 'class' 'probabilities' format \n        # 'probabilites' is floating point number between 0 and 1. \n    print(categories[top5_catid[i]], top5_prob[i].item())\n")),(0,o.kt)("h2",{id:"the-output"},"The output"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"Samoyed 0.8847386837005615\nArctic fox 0.045728735625743866\nwhite wolf 0.04426601156592369\nPomeranian 0.005614196881651878\nGreat Pyrenees 0.00464773690328002\n")),(0,o.kt)("p",null,"This mean our ",(0,o.kt)("inlineCode",{parentName:"p"},"dog.jpg")," is likely to be Samoyed!"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Acknowledgement"),": The content of this document has been adapted from the original ",(0,o.kt)("a",{parentName:"p",href:"https://pytorch.org/hub/pytorch_vision_resnet"},"PyTorch ResNet"),"."))}d.isMDXComponent=!0}}]);