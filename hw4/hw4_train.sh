#!/usr/bin/env bash



wget --no-check-certificate 'https://www.dropbox.com/s/426gg2xjp4qpqsm/emb.zip?dl=0' -O emb.zip
unzip emb.zip


python3 hw4_train.py $1 $2 ./emb
