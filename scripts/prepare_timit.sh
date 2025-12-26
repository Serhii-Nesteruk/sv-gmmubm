#!/bin/bash

find ../data/timit/ -type f \( \
  -name "SA*" \
  -o -name "SX*.WAV" \
  -o -name "SI*.WAV" \
  -o -name "*.TXT" \
  -o -name "*.txt" \
  -o -name "*.csv" \
  -o -name "*.DOC" \
  -o -name "*.WRD" \
  -o -name "*.PHN" \
\) -delete && \
mv ../data/timit/data/TEST ../data/timit/TEST && \
mv ../data/timit/data/TRAIN ../data/timit/TRAIN && rm -rf ../data/timit/data