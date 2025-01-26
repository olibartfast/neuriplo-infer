#!/bin/bash

LIBTORCH_VERSION=2.0.0
LIBTORCH_DIR=$HOME/libtorch-linux-x64-shared-with-deps-$LIBTORCH_VERSION

# Compute platform from pytorch.org, i.e. cu118, cu121, rocm6.0 or cpu
COMPUTE_PLATFORM=cpu 
wget https://download.pytorch.org/libtorch/$COMPUTE_PLATFORM/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION%2B$COMPUTE_PLATFORM.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION+$COMPUTE_PLATFORM.zip -d $HOME && \
    rm libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION+$COMPUTE_PLATFORM.zip
