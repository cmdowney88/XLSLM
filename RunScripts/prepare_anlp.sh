#!/bin/sh
source activate slm

python Tools/prepare_americas_nlp.py \
    Data/AmericasNLP/Aymara/train.aym \
    Data/AmericasNLP/Aymara/dev.aym \
    Data/AmericasNLP/Bribri/train.bzd \
    Data/AmericasNLP/Bribri/dev.bzd \
    Data/AmericasNLP/Ashaninka/train.cni \
    Data/AmericasNLP/Ashaninka/dev.cni \
    Data/AmericasNLP/Guarani/train.gug \
    Data/AmericasNLP/Guarani/dev.gug \
    Data/AmericasNLP/Wixarika/train.hch \
    Data/AmericasNLP/Wixarika/dev.hch \
    Data/AmericasNLP/Nahuatl/train.nah \
    Data/AmericasNLP/Nahuatl/dev.nah \
    Data/AmericasNLP/Hñähñu/train.oto \
    Data/AmericasNLP/Hñähñu/dev.oto \
    Data/AmericasNLP/Quechua/train.quy \
    Data/AmericasNLP/Quechua/dev.quy \
    Data/AmericasNLP/ShipiboKonibo/train.shp \
    Data/AmericasNLP/ShipiboKonibo/dev.shp \
    Data/AmericasNLP/Raramuri/train.tar \
    Data/AmericasNLP/Raramuri/dev.tar
