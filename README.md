# Crystallography Companion Agent (xca)

The application of this package is demonstrated in [aXiv:2008.00283](https://arxiv.org/abs/2008.00283).

**ABSTRACT:** The discovery of new structural and functional materials is driven by phase identification, often using X-ray diffraction (XRD). Automation has accelerated the rate of XRD measurements, greatly outpacing XRD analysis techniques that remain manual, time consuming, error prone, and impossible to scale. With the advent of autonomous robotic scientists or self-driving labs, contemporary techniques prohibit the integration of XRD. Here, we describe a computer program for the autonomous characterization of XRD data, driven by artificial intelligence (AI), for the discovery of new materials. Starting from structural databases, we train an ensemble model using a physically accurate synthetic dataset, which output probabilistic classifications --- rather than absolutes --- to overcome the overconfidence in traditional neural networks. This AI agent behaves as a companion to the researcher, improving accuracy and offering unprecedented time savings, and is demonstrated on a diverse set of organic and inorganic materials challenges. This innovation is directly applicable to inverse design approaches, robotic discovery systems, and can be immediately considered for other forms of characterization such as spectroscopy and the pair distribution function.

Dataset generation makes extensive use of the [cctbx](https://cctbx.github.io/), which can be setup in a ipython kernel following [these instructions](https://medium.com/@sljack1992/making-a-custom-ipython-notebook-kernel-c59e493de0b6).
The cctbx still uses python 2, so the dataset generation (python 2) is separated from the machine learning (python 3). 

*This repository is in the process of being ported. Please contact pmm for further information.* 
