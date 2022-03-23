#!/usr/bin/env bash

PREFIX="../data/raw/blur80"
rm -rf "${PREFIX}"

# LOAD SUN 80 DATASET WITH KERNELS
LINK="1LxIQcspBUaXE7-IROM4c-ImTFJbrxmLk"
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${LINK}" -P $PREFIX  
mv "$PREFIX/uc?export=download&id=${LINK}" "${PREFIX}/blur80.tar.xz"
tar -xf "${PREFIX}/blur80.tar.xz" --directory ${PREFIX}

IMAGES="${PREFIX}/blur_80/clean_images"

# SPLIT ONLY 5 IMAGES FOR TEST
./split.py $IMAGES "${PREFIX}/blur_80/images" -n 5

# PROCESS DATASET
IMAGES="${PREFIX}/blur_80/images/test"
KERNELS="${PREFIX}/blur_80/kernels"

OUTPUT_DIR="../data/blur80/"
GT="${OUTPUT_DIR}/gt"
BLURRED="${OUTPUT_DIR}/blurred"
NOISED="${OUTPUT_DIR}/noised"

./process.py $IMAGES $GT
mkdir "${OUTPUT_DIR}/kernels"

let i=0
for file in $(ls $KERNELS)
do
    echo "Processing $file..."
    echo "${KERNELS}/${file}"
    ./process.py $IMAGES "$BLURRED/kernel_${i}" -b --kernel_path "${KERNELS}/${file}"
    for std in 0.01 0.05
    do
        ./process.py "$BLURRED/kernel_${i}" "$NOISED/${std}/kernel_${i}" -s $std
    done
    cp "${KERNELS}/${file}" "${OUTPUT_DIR}/kernels/kernel_${i}.png"
    let i=$i+1
done

rm -rf "${PREFIX}"