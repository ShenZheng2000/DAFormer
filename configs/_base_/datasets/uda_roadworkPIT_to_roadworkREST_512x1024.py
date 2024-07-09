# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (512, 1024) # NOTE: remove cropping for now
img_scale = (1024, 512)
roadwork_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # Only include if annotations are available
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),  # Ensures all images are resized to 1024x512
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
# TODO: prepare train_ALL, train_PIT and train_REST, test_REST
# NOTE: for this roadwork
root = '/longdata/anurag_storage/workzone_segm/sem_seg'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='ROADWORKDataset',
            data_root=root,
            img_dir='images/train_PIT',
            ann_dir='gtFine/train_PIT',
            pipeline=roadwork_train_pipeline),
        target=dict(
            type='ROADWORKDataset',
            data_root=root,
            img_dir='images/train_REST',
            ann_dir='gtFine/train_REST',
            pipeline=roadwork_train_pipeline)),
    val=dict(
        type='ROADWORKDataset',
        data_root=root,
        img_dir='images/test_REST',
        ann_dir='gtFine/test_REST',
        pipeline=test_pipeline),
    test=dict(
        type='ROADWORKDataset',
        data_root=root,
        img_dir='images/test_REST',
        ann_dir='gtFine/test_REST',
        pipeline=test_pipeline))
