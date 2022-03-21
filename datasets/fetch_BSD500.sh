#!/usr/bin/env bash

PREFIX="../data/raw"
rm -rf "${PREFIX}"

# RAW DOWNLOADED DATA
TRAIN_DATA_NAME="${PREFIX}/BSR/BSDS500/data/images/train"
VAL_DATA_NAME="${PREFIX}/BSR/BSDS500/data/images/val"
TEST_DATA_NAME="${PREFIX}/BSR/BSDS500/data/images/test"

# LOAD BSD500 DATASET
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -P ../data/raw
tar -xvzf "${PREFIX}/BSR_bsds500.tgz" --directory ${PREFIX}


#  DATA
TRAIN_PROCESSED="../data/train"
TEST_PROCESSED="../data/test"

rm -rf $TRAIN_PROCESSED
rm -rf $TEST_PROCESSED

mkdir -p $TRAIN_PROCESSED
mkdir -p $TEST_PROCESSED


# TODO 
# run python processing from script