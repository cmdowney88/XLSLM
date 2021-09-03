#!/bin/sh
source activate slm

python Tools/file_stats.py \
    Data/AmericasNLP/Aymara/train.aym \
    Data/AmericasNLP/Aymara/dev.aym \
    Data/AmericasNLP/Bribri/train.bzd \
    Data/AmericasNLP/Bribri/dev.bzd \
    Data/AmericasNLP/Ashaninka/train.cni \
    Data/AmericasNLP/Ashaninka/dev.cni \
    Data/AmericasNLP/Ashaninka/train.mono.cni \
    Data/AmericasNLP/Ashaninka/dev.mono.cni \
    Data/AmericasNLP/Guarani/train.gug \
    Data/AmericasNLP/Guarani/dev.gug \
    Data/AmericasNLP/Wixarika/train.hch \
    Data/AmericasNLP/Wixarika/dev.hch \
    Data/AmericasNLP/Nahuatl/train.nah \
    Data/AmericasNLP/Nahuatl/dev.nah \
    Data/AmericasNLP/Hñähñu/train.oto \
    Data/AmericasNLP/Hñähñu/dev.oto \
    Data/AmericasNLP/Quechua/train.quy \
    Data/AmericasNLP/Quechua/train_downsampled.quy \
    Data/AmericasNLP/Quechua/dev.quy \
    Data/AmericasNLP/ShipiboKonibo/train.shp \
    Data/AmericasNLP/ShipiboKonibo/dev.shp \
    Data/AmericasNLP/ShipiboKonibo/train.mono.shp \
    Data/AmericasNLP/ShipiboKonibo/dev.mono.shp \
    Data/AmericasNLP/Raramuri/train.tar \
    Data/AmericasNLP/Raramuri/dev.tar \
    Data/AmericasNLP/train.anlp \
    Data/AmericasNLP/train_balanced.anlp \
    Data/AmericasNLP/dev.anlp
