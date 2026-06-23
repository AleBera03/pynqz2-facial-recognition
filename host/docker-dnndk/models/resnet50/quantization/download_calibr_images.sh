#!/bin/bash

OUTPUT_DIR="./images"
if [[ ! -d $OUTPUT_DIR ]]; then
  mkdir -p $OUTPUT_DIR
fi

curl -L -o ./face-detection-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/fareselmenshawii/face-detection-dataset

unzip ./face-detection-dataset.zip ./face-detection-dataset

cnt=0
for file in ./face-detection-dataset/images/train/* ; do
  cp $file "$OUTPUT_DIR/$cnt.jpg"
  $((cnt++))
  if [[ $cnt -eq 1000 ]]; then
    break
  fi
done

rm -rf ./face-detection-dataset