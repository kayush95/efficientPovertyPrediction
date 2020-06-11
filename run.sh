#!/bin/bash 

python pretrain_seq.py --model policy_satellite \
    --lr 1e-4 \
    --cv_dir cv/normal \
    --batch_size 1 \
    --data_dir efficient_17X17 \
    --parallel

