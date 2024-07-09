# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class GMDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(GMDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
