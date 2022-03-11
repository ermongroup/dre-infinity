#!/bin/bash

for i in 1 2 3 4 5
do
  echo $i
  CUDA_VISIBLE_DEVICES=0 python3 experiments/images_centering_copula.py eval_on_test \
    with experiments/image_configs/copula/${i}.json
done