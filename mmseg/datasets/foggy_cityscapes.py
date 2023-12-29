# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyCityscapesDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(FoggyCityscapesDataset, self).__init__(
            img_suffix='leftImg8bit_foggy_beta_0.02.png', # NOTE: pick the heavy foggy images
            seg_map_suffix='gtFine_labelTrainIds.png',
            **kwargs)
