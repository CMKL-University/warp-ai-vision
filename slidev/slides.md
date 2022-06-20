---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: https://source.unsplash.com/collection/94734566/1920x1080
background: /sandy-bg.jpeg
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: false
# some information about the slides, markdown enabled
info: |
  AiCE Warp AI Computer Vision Lectures
# persist drawings in exports and build
drawings:
  persist: false
---

# AiCE Warp
## Computer Vision and its Applications

Dr. Akkarit Sangpetch<br/>
AiCE Program Director, CMKL University

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
Introduction to AI and computer vision
-->

---

# Overview

For this session, we'll guide you through the following materials.
- 🧑‍💻 **Instructors** - get to know your coach and TAs
- 🏞 **What is Computer Vision?** - a little bit about CV and its application
- 📝 **Applications** - applications of computer vision
- 🗂 **Tools & Tasks** - toolings & how to get started

## Acknowledgement
Our slides have been composed together from many different people and materials. Special thanks to the following people for making their lecture notes and materials available online: Kris Kitani, Bob Collins, Srinivasa Narashiman, Martial Hebert, Alyosha Efros, Ali Faharadi, Deva Ramanan, Yaser Sheikh, Todd Zickler, Ioannis Gkioulekas, Steve Seitz, Richard Szeliski, Larry Zitnick, Noah Snavely, Lana Lazebnik, Kristen Grauman, Yung-Yu Chuang, Tinne Tuytelaars, Fei-Fei Li, Antonio Torralba, Rob Fergus, David Claus, Nick Babich, and Dan Jurafsky.

<!--
Program Overview
-->

---

# Instructors

<div grid="~ cols-3 gap-4">

![Remote Image](https://github.com/akkarit.png)

![Remote Image](https://github.com/paleumm.png)

![Remote Image](https://github.com/seksu.png)

Akkarit Sangpetch<br/>Instructor<br/>`@akkarit`

Permpoon Boonyarit<br/>Teaching Assistant (Lab)<br/>`@paleumm`

Jarukit Suchat<br/>Teaching Assistant (Tech Support)<br/>`@seksu`

</div>

---
layout: center
class: text-center
---

# What is Computer Vision?

---
layout: cover
background: /yosemite.jpeg
---

# What do you see?

<h2 v-click>Why are you able to interpret the image?</h2>

---

# What does a computer see?

![Local Image](/lincoln_pixel_values.png)
**Left:** Digital image; **Center:** Pixels labeled with a number from 0-255; **Right:** Image representation [^1]

[^1]: [Image credit: openframeworks](https://openframeworks.cc/ofBook/chapters/image_processing_computer_vision.html)

<!-- Image credit: https://openframeworks.cc/ofBook/chapters/image_processing_computer_vision.html -->

---
layout: image-right
image: /computer-vision.png
---

# What is Computer Vision?
Computer vision is a field in computing that focuses on creating a digital system that can process, analyze, and make sense of visual data.
- Given input images/videos (representation)
- Processing (math alert!)
- Output (interpretation or decision)

<!--
Image source: https://industrywired.com/the-era-of-computer-vision-is-here/
-->

---
layout: center
class: text-center
---

# Goal of Computer Vision
## To give computer (super) human-level perception

---
layout: center
---

# Perception Pipeline
<img src="/human-and-artificial-sensing.png" class="h-80 rounded shadow" />

Human Vision & Computer Vision System [^1]

[^1]: [Image credit: Manning](https://freecontent.manning.com/mental-model-graphic-grokking-deep-learning-for-computer-vision/)

---
layout: center
class: text-center
---

# Applications of Computer Vision

---

# Autonomous Retail

<Youtube id="yeS8TJwBAFs" class="w-720px h-400px"/>
https://youtu.be/o-FDVhkjk1M
---

# Industrial Inspection

<Youtube id="o-FDVhkjk1M" class="w-720px h-400px"/>

---

# Face Detection

![Local Image](/face-crowd.jpg)

---

# Detect Plant Diseases

<Youtube id="90SY5wAZdbc" class="w-720px h-400px"/>

---

# Image Segmentation

![Local Image](/image-masks.jpeg)

<!-- Image source: https://pixellib.readthedocs.io/en/latest/Image_instance.html -->
---

# Simultaneous Localization and Mapping

<Youtube id="34n1tF5OtQU" class="w-720px h-400px"/>

---

# Self-driving Car

<Youtube id="GB4p_fjQZNE" class="w-720px h-400px"/>

---

# Tools & Tasks

Ok, that's quite a lot. Where do I get started?

- 🐍 **Python** - Another programming language? [Learn](https://docs.microsoft.com/en-us/learn/paths/python-language/)
- 💻 **IDE** - Try [Visual Studio Code](https://code.visualstudio.com/docs/languages/python) or [DataSpell](https://www.jetbrains.com/dataspell/)
- 🧮 **NumPy** - Get familiar with arrays & matrix manipulations; see [NumPy Tutorial](https://numpy.org/devdocs/user/quickstart.html)

If you've already got the basics:

- 📝 **Tutorial** - Head over to our [tutorial contents](https://cmkl-university.github.io/warp-ai-vision/) (we'll cover them in labs)
- 🧿 **Seeing AI** - Get together in groups & think about building your project over the next 5 weeks

---

# Recommended Reading

Richard Szeliski, Computer Vision: Algorithms and Applications, 2nd ed.

http://szeliski.org/Book/

![Remote Image](http://szeliski.org/Book/imgs/Szeliski2ndBookFrontCover.png)

---
layout: center
class: text-center
---

# Self-study Contents

[Tutorials](https://cmkl-university.github.io/warp-ai-vision/) · [GitHub](https://github.com/CMKL-University/warp-ai-vision)