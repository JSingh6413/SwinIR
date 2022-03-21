#!/usr/bin/env bash

PREFIX="../data/raw"
rm -rf "${PREFIX}"

# LOAD BSD500 DATASET
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -P ../data/raw
tar -xvzf "${PREFIX}/BSR_bsds500.tgz" --directory ${PREFIX}

#  MERGE DATASET
TRAIN_DATA="${PREFIX}/BSR/BSDS500/data/images/train"
VAL_DATA="${PREFIX}/BSR/BSDS500/data/images/val"
TEST_DATA="${PREFIX}/BSR/BSDS500/data/images/test"

MERGED="${PREFIX}/merged" 
./merge.py $TRAIN_DATA $VAL_DATA $TEST_DATA -o $MERGED

# TRAIN-TEST SPLIT
./split.py $MERGED "../data" -s -r 42 -n 68
rm -rf "${PREFIX}"