#!/bin/bash

MODEL_NAME="face_resnet50"
BUILD_FOLDER="../build"
SRC_MODEL_FOLDER="../quantization/quantize_results"

# delete previous results
rm -rf $BUILD_FOLDER
mkdir -p $BUILD_FOLDER/debug $BUILD_FOLDER/normal

# Compile
echo "#####################################"
echo "COMPILE WITH DNNC"
echo "#####################################"

dnnc \
       --parser=caffe \
       --prototxt="$SRC_MODEL_FOLDER/deploy.prototxt" \
       --caffemodel="$SRC_MODEL_FOLDER/deploy.caffemodel" \
       --dcf=pynqz2_dpu.dcf \
       --cpu_arch=arm32 \
       --output_dir=../build/debug \
       --save_kernel \
       --mode debug \
       --net_name=$MODEL_NAME

dnnc \
       --parser=caffe \
       --prototxt="$SRC_MODEL_FOLDER/deploy.prototxt" \
       --caffemodel="$SRC_MODEL_FOLDER/deploy.caffemodel" \
       --dcf=pynqz2_dpu.dcf \
       --cpu_arch=arm32 \
       --output_dir=../build/normal \
       --save_kernel \
       --mode normal \
       --net_name=$MODEL_NAME


echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"
