## DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation

## Training

<!-- For convenience, we provide an [annotated config file](configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py) of the final DAFormer.
A training job can be launched using:

```shell
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
```

For the experiments in our paper (e.g. network architecture comparison,
component ablations, ...),  -->

We use a system to automatically generate
and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.


<!-- 
## Checkpoints

Below, we provide checkpoints of DAFormer for different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [DAFormer for GTA→Cityscapes](https://drive.google.com/file/d/1pG3kDClZDGwp1vSTEXmTchkGHmnLQNdP/view?usp=sharing)
* [DAFormer for Synthia→Cityscapes](https://drive.google.com/file/d/1V9EpoTePjGq33B8MfombxEEcq9a2rBEt/view?usp=sharing)
* [DAFormer for Cityscapes→ACDC](https://drive.google.com/file/d/16RSBkzJbGprWr04LjyNleqRzRZgCaEBn/view?usp=sharing)
* [DAFormer for Cityscapes→DarkZurich](https://drive.google.com/file/d/1_VXKDhnp4x4sslBj5B8tqqBJXeOuI9hS/view?usp=sharing)
* [DAFormer for GTA Domain Generalization](https://drive.google.com/file/d/1up9x3R3HtU_MjM6F89xNIHzPbIqBSacx/view?usp=sharing)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.
* The logs provide the mIoU on the validation set. For Cityscapes→ACDC and
  Cityscapes→DarkZurich the results reported in the paper are calculated on the
  test split. For DarkZurich, the performance significantly differs between
  validation and test split. Please, read the section above on how to obtain
  the test mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for DAFormer are:

* [configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py](configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py):
  Annotated config file for the final DAFormer.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of UDA self-training with ImageNet Feature Distance.
* [mmseg/datasets/uda_dataset.py](mmseg/datasets/uda_dataset.py):
  Data loader for UDA with Rare Class Sampling.
* [mmseg/models/decode_heads/daformer_head.py](mmseg/models/decode_heads/daformer_head.py):
  Implementation of DAFormer decoder with context-aware feature fusion.
* [mmseg/models/backbones/mix_transformer.py](mmseg/models/backbones/mix_transformer.py):
  Implementation of Mix Transformer encoder (MiT).

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)

## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters. -->
