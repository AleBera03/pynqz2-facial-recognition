#!/bin/bash

OUTPUT_DIR="./quantize_results"
MODEL="resnet50_scratch"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
python gen_calibr_file.py

# check if images folder exists, if not just download it
if [[ ! -d "./images" ]]; then
  ./download_calibr_images.sh
fi

. ../../.env

# run quantization
echo "#####################################"
echo "QUANTIZE"
echo "#####################################"

if [[ ${MODE} == "cpu" ]]; then
  decent-cpu quantize \
  -output_dir $OUTPUT_DIR \
  -model $MODEL/$MODEL.prototxt \
  -weights $MODEL/$MODEL.caffemodel \
  -method 1
else
  decent quantize \
  -output_dir $OUTPUT_DIR \
  -model $MODEL/$MODEL.prototxt \
  -weights $MODEL/$MODEL.caffemodel \
  -method 1 \
  -gpu 0
fi

if [[ $? -ne 0 ]]; then
  echo "#####################################"
  echo "QUANTIZATION FAILED"
  echo "#####################################"
else
  echo "#####################################"
  echo "QUANTIZATION COMPLETED"
  echo "#####################################"

