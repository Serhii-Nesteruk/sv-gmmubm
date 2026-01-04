#!/bin/bash
set -e

mkdir -p ../data/features/TRAIN \
         ../data/features/TEST \
         ../data/models \
         ../data/timit

wget -P ../data/timit \
    http://www.openslr.org/resources/12/train-clean-100.tar.gz

tar -xzf ../data/timit/train-clean-100.tar.gz -C ../data/timit

rm -f ../data/timit/train-clean-100.tar.gz
