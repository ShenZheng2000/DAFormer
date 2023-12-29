# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

from ...transforms.fovea import build_grid_net, before_train_json, process_mmseg, read_seg_to_det

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 # NOTE: configs for warping
                 VANISHING_POINT=None, 
                 warp_aug=None,
                 warp_aug_lzu=None, 
                 warp_fovea=None, 
                 warp_fovea_inst=None,
                 warp_fovea_inst_scale=False,
                 warp_fovea_inst_scale_l2=False,
                 warp_fovea_mix=None, 
                 warp_middle=None,
                 warp_debug=False,
                 warp_fovea_center=False,
                 warp_scale=1.0,
                 warp_dataset=[],
                 SEG_TO_DET=None,
                 keep_grid=False,
                 is_seg=True,
                 bandwidth_scale=64,
                 amplitude_scale=1.0,
                 ):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        # NOTE: add these stuffs used for warping
        self.warp_aug = warp_aug
        self.warp_aug_lzu = warp_aug_lzu
        self.warp_fovea = warp_fovea
        self.warp_fovea_inst = warp_fovea_inst
        self.warp_fovea_mix = warp_fovea_mix
        self.warp_middle = warp_middle
        self.warp_debug = warp_debug
        self.warp_scale = warp_scale
        self.warp_dataset = warp_dataset
        self.warp_fovea_center = warp_fovea_center
        self.is_seg = is_seg
        self.keep_grid = keep_grid

        self.seg_to_det = read_seg_to_det(SEG_TO_DET)

        self.vanishing_point = before_train_json(VP=VANISHING_POINT)
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

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, img_metas=None, is_training=True):
        """Extract features from images."""

        # TODO: check later if img_metas is updated in-place
        # using warping image visualization, and print-outs

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
            else:
                x = self.backbone(img)
        else:
            x = self.backbone(img)

        if self.with_neck:
            x = self.neck(x)

        # return x, img, img_metas
        return x

    def encode_decode(self, img, img_metas, 
                      is_training=True
                      ):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # print("Before; encode_decode img.shape", img.shape)
        # x, img, img_metas = self.extract_feat(img, img_metas, 
        #                                       is_training
        #                                       )
        x = self.extract_feat(img, img_metas, 
                                is_training
                                )
        # print("After; encode_decode img.shape", img.shape)
        out = self._decode_head_forward_test(x, img_metas)
        
        # print("out shape", out.shape) # [1, 19, 128, 128]

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      is_training=True
                      ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x, img, img_metas = self.extract_feat(img, img_metas, 
        #                                       is_training
        #                                       )
        x = self.extract_feat(img, img_metas, 
                            is_training
                            )

        losses = dict()
        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, 
                        is_training=False,
                        ):
        """Inference with full image."""

        # TODO: hardcode this as True for now, set to False later
        # is_training = True

        seg_logit = self.encode_decode(img, img_meta, 
                                       is_training
                                       )

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
