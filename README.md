# Training

We use a system to automatically generate
and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.


# Warping Code Locations

1. Navigate to `mmseg/models/uda/dacs.py`

<details>
  <summary>Init No-Warp for Target</summary>
  <pre>
self.warp_tgt = cfg.get('warp_tgt', True)
  </pre>
</details>


<details>
  <summary>First No-Warp for Target</summary>
  <pre>
ema_logits = self.get_ema_model().encode_decode(
    target_img, target_img_metas,
    is_training=self.warp_tgt # NOTE: add non-warp for here
    ) 
  </pre>
</details>

<details>
  <summary>Second No-Warp for Target</summary>
  <pre>
mix_losses = self.get_model().forward_train(
  mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True, 
  is_training=self.warp_tgt # NOTE: add no warp flag here!!!
  ) 
  </pre>
</details>



2. Navigate to `mmseg/models/segmentors/encoder_decoder.py`

<details>
  <summary>Import Warping Modules</summary>
  <pre>
from ...transforms.fovea import build_grid_net, before_train_json, process_mmseg, read_seg_to_det
  </pre>
</details>

<details>
  <summary>Build Warping Grid</summary>
  <pre>
self.grid_net = build_grid_net(warp_aug_lzu=warp_aug_lzu,
                                warp_fovea=warp_fovea,
                                warp_fovea_inst=warp_fovea_inst,
                                warp_fovea_mix=warp_fovea_mix,
                                warp_middle=warp_middle,
                                warp_scale=warp_scale,
                                warp_fovea_center=warp_fovea_center,
                                warp_fovea_inst_scale=warp_fovea_inst_scale,
                                warp_fovea_inst_scale_l2=warp_fovea_inst_scale_l2,
                                is_seg=is_seg,
                                bandwidth_scale=bandwidth_scale,
                                amplitude_scale=amplitude_scale,)
  </pre>
</details>

<details>
  <summary>Warp Images and Unwarp Features</summary>
  <pre>
if (self.warp_aug_lzu is True) and (img_metas is not None):
    # print("self.warp_dataset is", self.warp_dataset)
    if any(src in img_metas[0]['filename'] for src in self.warp_dataset) and (is_training is True):
        # print(f"YES, RUNNING warping on {img_metas[0]['filename']}")
        x, img, img_metas = process_mmseg(img_metas,
                                            img,
                                            self.warp_aug_lzu,
                                            self.vanishing_point,
                                            self.grid_net,
                                            self.backbone,
                                            self.warp_debug,
                                            seg_to_det=self.seg_to_det,
                                            keep_grid=self.keep_grid
                                        )
        # print("images.shape", images.shape)
  </pre>
</details>

# Checkpoints

Download checkpoints from [[here](https://drive.google.com/drive/folders/1W9aMHqUbr34FB0TTaBrTGpgUMHqI9E7_?usp=drive_link)]

# Specific Configs

<details>
  <summary>Click Here</summary>

## Cityscapes -> DarkZurich

| Experiments | Id |
|----------|----------|
| DAFormer                | 80 |
| DAFormer + Sta. Prior   | 83 |
| DAFormer + Geo. Prior   | 84 |
| DAFormer + Ours         | 88 |

## Cityscapes -> ACDC

| Experiments | Id |
|----------|----------|
| DAFormer                | 90 |
| DAFormer + Sta. Prior   | 93 |
| DAFormer + Geo. Prior   | 94 |
| DAFormer + Ours         | 98 |

## Cityscapes -> Foggy Cityscapes

| Experiments | Id |
|----------|----------|
| DAFormer                | 260 |
| DAFormer + Ours         | 268 |

## GTA -> Cityscapes

| Experiments | Id |
|----------|----------|
| DAFormer                | 210 |
| DAFormer + Ours         | 215 |

## Synthia -> Cityscapes

| Experiments | Id |
|----------|----------|
| DAFormer                | 220 |
| DAFormer + Ours         | 228 |

</details>