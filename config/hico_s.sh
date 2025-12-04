#!/usr/bin/env bash

set -x
EXP_DIR = output_path

python  main.py \
        --pretrained params/HICO_GEN_VLKT_S_21.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data_path \
        --num_obj_classes 22 \
        --num_verb_classes 10 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers 3 \
        --mamba_layers 6 \
        --epochs 30 \
        --use_nms_filter \
        --mimic_loss_coef 20 \
        --batch_size 4 \
        --lr 2e-4 \
        --lr_drop 10 \
        --lr_mamba 1e-4 \
        --with_mimic \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
