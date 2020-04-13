# Reimplementation of Modeling Nonlinear Audio Effects with End-to-end Deep Neural Networks 
Experimental implementation of [Modeling Nonlinear Audio Effects with End-to-end Deep Neural Networks.](https://ieeexplore.ieee.org/abstract/document/8683529/) [Original work](https://github.com/mchijmma/DL-AFx) by [Marco Martínez](https://github.com/mchijmma)


# WARNING
This repository is incomplete. It was a reimplementation before the release of Marco's journal paper and accompanying code for all of the end-to-end analog audio effect models:  https://github.com/mchijmma/DL-AFx

### TODO
- Smooth Adaptive Activation Function (SAAF) - with an adaptive tanh on the backend this model produces artefacts.

## Requirements
`Python 3.6.6`
`CUDNN`
`keras 2.1.6`
`tensorflow-gpu 1.12.0`

## bash config
For a consisent environment configuration, add the following to your `~/.bashrc`

```
module load python/3.6.6

export PATH=${PATH}:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

module load cuda/10.0-cudnn7.4.2

export CUDA_VISIBLE_DEVICES=0
```

-------------- WIP --------------
