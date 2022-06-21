---
sidebar_position: 1
---

# Setting Up

Let's try to setup your environment.

## Getting Started

Get started by **[installing Conda](https://docs.anaconda.com/anaconda/install/)**.
If you feel like running experimental software, **[mamba](https://github.com/conda-forge/miniforge#mambaforge)** is a faster replacement but not as popular as conda.

## Create an environment

Create a new working environment named `warp`.
```bash
conda create -n warp python=3
conda activate warp
```
You can type the above commands into Command Prompt, Powershell, or any other integrated terminal of your code editor. If you are still looking for one, try **[warp](https://www.warp.dev/)** and **[starship](https://starship.rs/)** :-) 

## Verify your environment
Verify that your environment has been setup.
```bash
conda list
```
You should get something similar to this. These are the initial list of packages that we'll use for our sessions.
```
# packages in environment at /Users/uname/mambaforge/envs/warp:
#
# Name                    Version                   Build  Channel
bzip2                     1.0.8                h620ffc9_4
ca-certificates           2022.4.26            hca03da5_0
certifi                   2022.5.18.1     py310hca03da5_0
libcxx                    12.0.0               hf6beb65_1
libffi                    3.4.2                hc377ac9_4
ncurses                   6.3                  h1a28f6b_2
openssl                   1.1.1o               h1a28f6b_0
pip                       21.2.4          py310hca03da5_0
python                    3.10.4               hbdb9e5c_0
readline                  8.1.2                h1a28f6b_1
setuptools                61.2.0          py310hca03da5_0
sqlite                    3.38.3               h1058600_0
tk                        8.6.12               hb8d0fd4_0
tzdata                    2022a                hda174b7_0
wheel                     0.37.1             pyhd3eb1b0_0
xz                        5.2.5                h1a28f6b_1
zlib                      1.2.12               h5a0b063_2
```
## Switching Projects

When you work on different projects, they may require different packages or dependencies. Conda (or mamba) can help you manage multiple package environments associated with each project.

Setting up a different project `elderwarp` with Python 3.9 and switch to the new environment.
```bash
conda create -n elderwarp python=3.9
conda activate elderwarp
```
After doing some experiment, you can then switch back to your previous environment.
```bash
conda deactivate
```
If necessary, you can also remove the environment you may no longer need.
```bash
conda env remove -n elderwarp
```
Check that you have removed the environment.
```bash
conda env list
```

## Setting up your IDE
We recommend setting up either of the following IDEs for your projects:
- Visual Studio Code: https://code.visualstudio.com/docs/python/python-tutorial
- DataSpell: https://www.jetbrains.com/dataspell/