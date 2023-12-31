# Training

We use a system to automatically generate
and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

# Specific Configs and Checkpoints

## Cityscapes -> DarkZurich

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| DAFormer                | 80 | TODO |
| DAFormer + Sta. Prior   | 83 | TODO |
| DAFormer + Geo. Prior   | 84 | TODO |
| DAFormer + Ours         | 88 | TODO |

## Cityscapes -> ACDC

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| DAFormer                | 90 | TODO |
| DAFormer + Sta. Prior   | 93 | TODO |
| DAFormer + Geo. Prior   | 94 | TODO |
| DAFormer + Ours         | 98 | TODO |

## Cityscapes -> Foggy Cityscapes

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| DAFormer                | 260 | TODO |
| DAFormer + Ours         | 268 | TODO |

## GTA -> Cityscapes

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| DAFormer                | 210 | TODO |
| DAFormer + Ours         | 215 | TODO |

## Synthia -> Cityscapes

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| DAFormer                | 220 | TODO |
| DAFormer + Ours         | 228 | TODO |