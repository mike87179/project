#!/usr/bin/env bash

wget --no-check-certificate 'https://www.dropbox.com/s/daqfxxay8825dfr/model.zip?dl=0' -O model.zip
unzip model.zip
rm model.zip

python3 hw3_test.py $1 $2 

