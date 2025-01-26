#!/bin/bash
TRT_MAJOR=10
TRT_MINOR=.7
TRT_PATCH=.0
TRT_BUILD=.23
TRT_VERSION=${TRT_MAJOR}${TRT_MINOR}${TRT_PATCH}${TRT_BUILD}
TRT_CUDA_VERSION=12.6
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_MAJOR}${TRT_MINOR}${TRT_PATCH}/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz 

tar -xvf TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz

mv TensorRT-${TRT_VERSION} $HOME/TensorRT-${TRT_VERSION}
