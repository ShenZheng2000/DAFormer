import os.path as osp
import mmcv
import numpy as np
from PIL import Image
from .builder import DATASETS
from .cityscapes import CityscapesDataset
from .builder import DATASETS

@DATASETS.register_module()
class ROADWORKDataset(CityscapesDataset):
    """ROADWORK dataset based on Cityscapes.

    This dataset uses customized classes and colors specific to roadwork scenes.
    """

    # TODO: if error persist with smaller gpu, try _Ids.png instead of _labelIds.png???
    
    CLASSES = (
        'unlabeled', 'Road', 'Sidewalk', 'Bike Lane', 'Off-Road', 'Roadside',
        'Barrier', 'Barricade', 'Fence', 'Police Vehicle', 'Work Vehicle',
        'Police Officer', 'Worker', 'Cone', 'Drum', 'Vertical Panel',
        'Tubular Marker', 'Work Equipment', 'Arrow Board', 'TTC Sign'
    )

    PALETTE = [
        [0, 0, 0],       # unlabeled
        [70, 70, 70],    # Road
        [102, 102, 156], # Sidewalk
        [190, 153, 153], # Bike Lane
        [180, 165, 180], # Off-Road
        [150, 100, 100], # Roadside
        [246, 116, 185], # Barrier
        [248, 135, 182], # Barricade
        [251, 172, 187], # Fence
        [255, 68, 51],   # Police Vehicle
        [255, 104, 66],  # Work Vehicle
        [184, 107, 35],  # Police Officer
        [205, 135, 29],  # Worker
        [30, 119, 179],  # Cone
        [44, 79, 206],   # Drum
        [102, 81, 210],  # Vertical Panel
        [170, 118, 213], # Tubular Marker
        [214, 154, 219], # Work Equipment
        [241, 71, 14],   # Arrow Board
        [254, 139, 32]   # TTC Sign
    ]

    def __init__(self, **kwargs):
        super(ROADWORKDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_labelIds.png',
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images with customized palette.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx", the png files will be named "somepath/xxx.png".
            to_label_id (bool): Whether to convert output to label_id for submission.

        Returns:
            list[str: str]: Resulting image files paths.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            palette = np.array(self.PALETTE, dtype=np.uint8).flatten().tolist()
            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def evaluate(self, results, metric='mIoU', logger=None, imgfile_prefix=None, efficient_test=False):
        """Evaluate the results using metrics like mIoU and Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file, for cityscapes evaluation only. If results are evaluated with cityscapes protocol, it would be the prefix of output png files. The output files would be png images under folder "a/b/prefix/xxx.png", where "xxx" is the image name of cityscapes. If not specified, a temp file will be created for evaluation. Default: None.

        Returns:
            dict[str, float]: Evaluation metrics.
        """
        eval_results = dict()
        metrics = metric if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if metrics:
            eval_results.update(super(ROADWORKDataset, self).evaluate(results, metrics, logger, efficient_test))

        return eval_results