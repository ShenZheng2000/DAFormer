# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (512, 1024) # NOTE: remove cropping for now
img_scale = (1024, 512)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
dark_zurich_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', img_scale=(960, 540)),  # original 1920x1080
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),  # NOTE: hardcode as cs shape now
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(960, 540),  # original 1920x1080
        img_scale=img_scale, # NOTE: hardcode as cs shape now
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
root = '/home/aghosh/Projects/2PCNet/Datasets'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root=f'{root}/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='DarkZurichDataset',
            data_root=f'{root}/dark_zurich/',
            img_dir='rgb_anon/train/night/',
            ann_dir='gt/train/night/',
            pipeline=dark_zurich_train_pipeline)),
    val=dict(
        type='DarkZurichDataset',
        data_root=f'{root}/dark_zurich/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline),
    test=dict(
        type='DarkZurichDataset',
        data_root=f'{root}/dark_zurich/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline))