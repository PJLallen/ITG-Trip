#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_r101_all_ivt_lr2e-4_nmsoutput_subfeature_traineval_noobjclip_detrbox

python  main.py \
        --pretrained params/HICO_GEN_VLKT_M_21.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data_path \
        --num_obj_classes 21 \
        --num_verb_classes 10 \
        --backbone resnet101 \
        --num_queries 100 \
        --dec_layers 3 \
        --epochs 50 \
        --lr_drop 20 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_mimic \
        --mimic_loss_coef 20 \
        --batch_size 4 \
        --lr 2e-4
        # --with_obj_clip_label \

