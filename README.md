# Training

We use a system to automatically generate
and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

# Checkpoints

Download checkpoints from [[here](https://drive.google.com/drive/folders/1W9aMHqUbr34FB0TTaBrTGpgUMHqI9E7_?usp=drive_link)]

# Specific Configs

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