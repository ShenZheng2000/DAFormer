# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import itertools
import logging
import math


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_half_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        # Setup optimizer and schedule
        if 'dacs' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)

        # NOTE: Add the custom hook for the best checkpoint based on mIoU, and every ckpts
        # cfg['checkpoint_config'] = dict(
        #     by_epoch=False, interval=iters, max_keep_ckpts=1)
        # cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters // 10, max_keep_ckpts=10)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU', save_best='mIoU')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if 'dacs' in uda and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    # TODO: hardcode bs as 1 for debug now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    batch_size = 2
    # batch_size = 1
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 4
    rcs_T = None
    plcrop = False


    # NOTE: rewrite the part above for simplicity
    # Define a dictionary to map dataset names
    DATASET_NAME_MAPPING = {
        'darkzurich': 'dark_zurich',
        'acdc': 'acdc',
        'synthia': 'synthia',
        'gta': 'gta',
        'cityscapes': 'cityscapes',
        'idd': 'idd',
        'foggy_cityscapes': 'foggy_cityscapes',
    }

    def get_warp_dataset_name(dataset_name):
        """Return the corresponding warp_dataset name for a given dataset name."""
        return DATASET_NAME_MAPPING.get(dataset_name, dataset_name)  # default to the original name if no mapping is found

    def config_for_id(id_value, src_dataset, tgt_dataset):
        if id_value % 10 == 0:
            return []

        src_warp_dataset_name = get_warp_dataset_name(src_dataset)
        tgt_warp_dataset_name = get_warp_dataset_name(tgt_dataset)

        cfg = {
            'model': {
                'VANISHING_POINT': None,
                # 'VANISHING_POINT': '/home/aghosh/Projects/2PCNet/Datasets/VP/cityscapes_to_darkzurich_all_vp.json',
                # 'VANISHING_POINT': "/home/aghosh/Projects/2PCNet/Datasets/VP/cs_dz_acdc_synthia_all_vp.json", # NOTE: replace with this ALL VP files
                'warp_aug_lzu': True,
                'warp_dataset': [src_warp_dataset_name, tgt_warp_dataset_name]
            },
        }

        # TODO: use this for visual debug only!
        cfg['model']['warp_debug'] = False

        if id_value in [81, 91]: # ======> fovea warping (src & tgt)
            cfg['model']['warp_fovea'] = True

        elif id_value in [82, 92]: # ======> tpp warping (src & tgt)
            pass  # Placeholder; add configs if needed
            cfg['model']['warp_fovea_center'] = True

        elif id_value in [83, 93]: # ======> fovea warping (src)
            cfg['model']['warp_fovea'] = True
            cfg.setdefault('uda', {})['warp_tgt'] = False

        elif id_value in [84, 94]: # ======> tpp warping (src)
            cfg['model']['warp_fovea_center'] = True
            cfg.setdefault('uda', {})['warp_tgt'] = False

        elif id_value in [85, 95]: # ======> bbox-level warping (pseudo)
            cfg['model']['warp_fovea_inst'] = True
            cfg.setdefault('uda', {})['warp_tgt'] = False
            # NOTE: reserve for gt bboxes before, no use for now
            cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/cityscapes_seg2det.json'

        else: # ======> bbox-level warping (src)
            cfg['model']['warp_fovea_inst'] = True
            cfg.setdefault('uda', {})['warp_tgt'] = False
            cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/cityscapes/gt_detection/instancesonly_filtered_gtFine_train_poly_simple.json'

            # 89, 99 => need to think about this later

            if id in [86, 96]:
                cfg['model']['bandwidth_scale'] = 32
            elif id in [87, 97,]:
                cfg['model']['bandwidth_scale'] = 128

            elif id in [101, 105,]:
                cfg['model']['amplitude_scale'] = 0.5
            elif id in [102, 106,]:
                cfg['model']['amplitude_scale'] = 0.75

            # our final model (something ends with 8) => no need to change anything

            # NOTE: add pseudo bbox warping configs here for synthia to cityscapes, and gta to cityscapes
            elif id == 215:
                cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/gta_seg2det.json'
            elif id == 225:
                cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/synthia_seg2det.json'
            elif id == 228:
                cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/synthia_seg2det_gt.json'
            elif id == 275:
                cfg['model']['SEG_TO_DET'] = '/home/aghosh/Projects/2PCNet/Datasets/IDD_seg2det.json'

            else:
                pass


        return cfg


    # Main code
    # if 80 <= id <= 269:
    if True:

        # Predefine datasets for experiments
        if 80 <= id <= 89 or 101 <= id <= 104: # 101-104 is the special case here
            datasets = [('cityscapes', 'darkzurich')]
        elif 110 <= id < 120 or 130 <= id < 140 or 150 <= id < 160 or 170 <= id <= 179 or 190 <= id <= 199 or 230 <= id <= 239:
            datasets = [('cityscapes', 'darkzurich')]  # for cs2dz with varying image percentages
        elif 120 <= id < 130 or 140 <= id < 150 or 160 <= id <= 169 or 180 <= id <= 189 or 200 <= id <= 209 or 240 <= id <= 249:
            datasets = [('cityscapes', 'acdc')]       # for cs2acdc with varying image percentages\
        # (id 210 to 219) => gta2cs experiments
        elif 210 <= id <= 219:
            datasets = [('gta', 'cityscapes')]
        # (id 220 to 229) => cs2syn experiments
        elif 220 <= id <= 229:
            datasets = [('synthia', 'cityscapes')]
        elif 250 <= id <= 259:
            datasets = [('cityscapes', 'idd')]
        elif 260 <= id <= 269:
            datasets = [('cityscapes', 'foggy_cityscapes')]
        elif 270 <= id <= 279:
            datasets = [('idd', 'cityscapes')]
        else:
            datasets = [('cityscapes', 'acdc')]

        seeds = [0]  # NOTE: keep seed = 0 for now
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        rcs_T = 0.01
        plcrop = True

        crop = '512x1024'

        # ############## Here is the dataset ratio for difference experiments ##############
        # 110+ => (25% of images) [cs2dz, cs2acdc]
        if 110 <= id <= 129:
            crop = '512x1024_025'
            iters = 10000

        # 130+ => (50% of images) [cs2dz, cs2acdc]
        elif 130 <= id <= 149:
            crop = '512x1024_050'
            iters = 20000
        
        # 150+ => (75% of images) [cs2dz, cs2acdc]
        elif 150 <= id <= 169:
            crop = '512x1024_075'
            iters = 30000

        elif 170 <= id <= 189:
            crop = '512x1024_050_uniform'
            iters = 20000

        elif 190 <= id <= 209:
            print("using uniform sampling!!!!!!!!!!!!!!!!!!!!")
            crop = '512x1024_025_uniform'
            iters = 10000

        elif 230 <= id <= 249:
            crop = '1024x2048'
            # NOTE: half batch_size, and double iters
            batch_size = 1
            iters = 80000

        for (source, target), seed in itertools.product(datasets, seeds):
            cfg = config_from_vars()  # base configuration
            # specific_cfg = config_for_id(id, target)
            specific_cfg = config_for_id(id, source, target)
            cfg.update(specific_cfg)  # Update base with specific configurations
            cfgs.append(cfg)

    return cfgs
