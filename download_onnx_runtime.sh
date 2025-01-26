#!/bin/bash
ORT_VERSION=1.19.2
ONNX_RUNTIME_DIR=$HOME/onnxruntime-linux-x64-gpu-$ORT_VERSION 

wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz && \
tar -xzvf onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz
mv onnxruntime-linux-x64-gpu-${ORT_VERSION} $ONNX_RUNTIME_DIR