#!/bin/bash

#CPU test
pytest --device=-1 --filepath=/data1/test ../tests/reader_test.py

#GPU test
for n in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )) ); do
    fpath=/data$(( 2 * $(( ${n} / 4 ))  + 1 ))/test
    #use module variant for env reasons:
    pytest --device=${n} --filepath=${fpath} ../tests/reader_test.py
done
