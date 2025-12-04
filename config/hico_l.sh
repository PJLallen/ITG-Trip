#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_gen_vlkt_l_r101_dec_6layers_feature_traineval_noobjclip

python  main.py \
        --pretrained params/HICO_GEN_VLKT_L_21.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data_path \
        --num_obj_classes 21 \
        --num_verb_classes 10 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers 6 \
        --epochs 50 \
        --lr_drop 20 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_mimic \
        --mimic_loss_coef 20 \
        --batch_size 4 \
        --lr 2e-4 \
