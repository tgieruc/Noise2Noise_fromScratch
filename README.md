# Noise2Noise, from scratch
[Colab](https://colab.research.google.com/github/tgieruc/Noise2Noise_fromScratch/blob/main/demo.ipynb)
• [Paper](https://arxiv.org/abs/1803.04189)

An implemenet of the [Noise2Noise](https://arxiv.org/abs/1803.04189) paper (Lehtinen et al., 2018), only using the PyTorch Tensor methods, with a working backpropagation.

## Data

I used 32x32 images from ImageNet with random gaussian noise. The images are available in the form of pickle files for
PyTorch. Pretrained weights are also available
on [this Drive](https://drive.google.com/drive/folders/198AnkC5XJtQgJ76DNL3v5iApzZdTPQ6_?usp=sharing).

## Use

A demo Jupyter Notebook is available [here](https://colab.research.google.com/github/tgieruc/Noise2Noise_fromScratch/blob/main/demo.ipynb).
