# Crystallography Companion Agent (xca)

## Overview
XCA is a psuedo-unsupervised learning approach for handling spherically integrated (powder) X-ray diffraction data.
The approach depends on accurate data synthesis that encompasses the multitude of abberations which can impact a diffraction pattern. 
The magnitude of influence of these aberrations is meant to be informed by experience given an experimental design (*e.g.* 
well prepared powders will experience significantly less texturing than epitaxially grown thin films.)
The dataset synthesis is accomplished using the [cctbx](https://cctbx.github.io/), starting from `.cif` files of potential phases. 
From synthetic datasets an ensemble of feed forward convolutional neural networks can be trained, and subsequently used
to predict phase existence in experimental data. 

# System requirements
## Hardware requirements
`xca` package requires only a standalone computer with enough RAM to support the in-memory operations.
For advanced use, a CUDA enabled GPU is recommended. 

## Software requirements
### OS requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:
- macOS: Catalina (10.15.7)
- Linux: Ubuntu 18.04

### Python dependencies 
`xca` Dataset generation makes extensive use of the [cctbx](https://cctbx.github.io/), 
which can be setup in a ipython kernel following [these instructions](https://medium.com/@sljack1992/making-a-custom-ipython-notebook-kernel-c59e493de0b6).
The cctbx still uses python 2, so the dataset synthesis (python 2) is separated from the machine learning in tensorflow (python 3). 

The machine learning depends on a scientific tensorflow stack: 
```
tensorflow >= 2.1.0
numpy
scikit-learn
scipy
``` 

## Installation guide
Due to mixed dependencies, we recommend  


# Getting started
## Literature Details 
The application of this package is demonstrated in [aXiv:2008.00283](https://arxiv.org/abs/2008.00283).

**ABSTRACT:** The discovery of new structural and functional materials is driven by phase identification, often using X-ray diffraction (XRD). Automation has accelerated the rate of XRD measurements, greatly outpacing XRD analysis techniques that remain manual, time consuming, error prone, and impossible to scale. With the advent of autonomous robotic scientists or self-driving labs, contemporary techniques prohibit the integration of XRD. Here, we describe a computer program for the autonomous characterization of XRD data, driven by artificial intelligence (AI), for the discovery of new materials. Starting from structural databases, we train an ensemble model using a physically accurate synthetic dataset, which output probabilistic classifications --- rather than absolutes --- to overcome the overconfidence in traditional neural networks. This AI agent behaves as a companion to the researcher, improving accuracy and offering unprecedented time savings, and is demonstrated on a diverse set of organic and inorganic materials challenges. This innovation is directly applicable to inverse design approaches, robotic discovery systems, and can be immediately considered for other forms of characterization such as spectroscopy and the pair distribution function.

