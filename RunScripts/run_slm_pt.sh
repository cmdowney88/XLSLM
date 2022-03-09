#!/bin/sh
source activate mslm

MODEL_NAME=$1
TRAIN_FILE=$2
PRETRAINED_MODEL=$3
DEV_CONFIG=$4

python3 -u -m mslm \
    --input_file=$TRAIN_FILE \
    --preexisting_model \
    --load_model_path=Models/${PRETRAINED_MODEL} \
    --model_path=Models/${MODEL_NAME} \
    --mode=train \
    --dev_config=$DEV_CONFIG \
    --config_file=Configs/${MODEL_NAME}.json
