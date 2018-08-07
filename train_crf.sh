#!/bin/bash


#根据CRF++的安装位置更改
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./CRF++-0.58/.lib
export PATH=$PATH:./CRF++-0.58


crf_learn -f 3 -c 4.0 template.txt output/data/train.txt  output/model/PKU.model