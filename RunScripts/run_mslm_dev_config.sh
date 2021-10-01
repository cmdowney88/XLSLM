#!/bin/bash
source activate mslm

MODEL_NAME=$1
TRAIN_FILE=$2
DEV_CONFIG=$3

python3 -u -m mslm \
    --input_file=$TRAIN_FILE \
    --model_path=Models/${MODEL_NAME} \
    --mode=train \
    --dev_config=DevConfigs/$DEV_CONFIG \
    --config_file=Configs/${MODEL_NAME}.json
