# Reimplementation of Modeling Nonlinear Audio Effects with End-to-end Deep Neural Networks 
Implementation of [Modeling Nonlinear Audio Effects with End-to-end Deep Neural Networks.](https://ieeexplore.ieee.org/abstract/document/8683529/) by Marco Martinez. [Original Implementation](https://github.com/mchijmma/DL-AFx) @ https://github.com/mchijmma

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

WIP 
