#!/bin/bash

cmake \
    -DCMAKE_CXX_COMPILER=g++ \
    -DPYTORCH_DIR=/opt/pytorch/pytorch \
    ..

make VERBOSE=1
