#!/bin/sh
source activate slm

python Tools/remove_duplicate_lines.py \
    Data/AmericasNLP/Nahuatl/train.nah \
    Data/AmericasNLP/Nahuatl/dev.nah \
    Data/KannEtAl2018/dev.nci

python Tools/remove_duplicate_lines.py \
    Data/AmericasNLP/Wixarika/train.hch \
    Data/AmericasNLP/Wixarika/dev.hch \
    Data/KannEtAl2018/dev.hch
