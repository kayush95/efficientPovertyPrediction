#!/bin/bash 
    
python test.py --model policy_satellite \
    --lr 1e-4 \
    --cv_dir cv/normal \
    --batch_size 1 \
    --data_dir efficient_17X17 \
    --parallel \
    --load cv/normal/ckpt_E_290_M_1.463_r2_0.619_R_3.71E-01_S_216.955_D_1518.060
