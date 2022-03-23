#! /usr/bin/env bash


INPUT_DIR="data/blur80/noised"
for std in 0.01 0.05
do
    for number in $(seq 8) 
    do
        let number=number-1
        INPUT_KERN="${INPUT_DIR}/${std}/kernel_${number}"
        OUTPUT_KERN="data/blur80/denoised/${std}/kernel_${number}"

        echo "Processing $INPUT_KERN..."
        ./eval_net.py $INPUT_KERN -o $OUTPUT_KERN --std ${std}    
    done
done