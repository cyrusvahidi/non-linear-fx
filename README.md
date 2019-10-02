# non-linear-fx
Implementation of Modelling non-linear audio effects with end-to-end deep learning.

## Requirements
`Python 3.6.6`
`CUDNN`

## bash config
For a consisent environment configuration, add the following to your `~/.bashrc` on the GPU server

```
module load python/3.6.6

export PATH=${PATH}:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

module load cuda/10.0-cudnn7.4.2

export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Essentially, this loads the cudnn library for tensorflow / torch and declares available GPUs for use.

## python virtual environment
Instructions coming soon for how to create a python virtual environment and which python packages to install.
