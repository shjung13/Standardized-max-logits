#!/usr/bin/env bash
# Example on Cityscapes
python -m torch.distributed.launch --nproc_per_node=1 calculate_statistics.py \
    --dataset cityscapes \
    --arch network.deepv3.DeepR101V3PlusD_OS8 \
    --city_mode 'train' \
    --lr_schedule poly \
    --lr 0.01 \
    --max_cu_epoch 10000 \
    --class_uniform_pct 0.5 \
    --class_uniform_tile 1024 \
    --poly_exp 0.9 \
    --snapshot ./pretrained/r101_os8_base_cty.pth \
    --crop_size 768 \
    --scale_min 0.5 \
    --scale_max 2.0 \
    --rrotate 0 \
    --max_iter 60000 \
    --bs_mult 4 \
    --gblur \
    --color_aug 0.5 \
    --date 0000 \
    --exp debug \
    --ckpt ./logs/ \
    --tb_path ./logs/

