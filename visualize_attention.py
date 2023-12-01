# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image



# NOTE: Set the random seed to a fixed number, for example, 42
torch.manual_seed(42)

# import utils
# import vision_transformer as vits
from mmseg.models.backbones.mix_transformer_change_11_5 import MixVisionTransformer # NOTE: use this one; otherwise model ckpts are wrong


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def print_direct_subkeys(state_dict):
    for k, v in state_dict.items():
        if isinstance(v, dict):
            print(f"Second-level keys for '{k}': {list(v.keys())}")


def load_and_transform_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = transform(img)
        return img
        
def process_images_recursively(root_dir):
    image_list = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is an image (e.g., .jpg, .png)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(subdir, file)
                image_tensor = load_and_transform_image(file_path)
                image_list.append((image_tensor, file_path))
    return image_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--attn_stage", type=int, default=1, help="Attention stage")
    parser.add_argument("--ref_index", type=int, default=0, help="Reference index")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # NOTE: change the model to DAFormer
    # build model
    # model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    model = MixVisionTransformer(patch_size=args.patch_size, 
                                 num_classes=0, # TODO: think about this later
                                 embed_dims=[64, 128, 320, 512])


    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    # print("model is", model)
    model.to(device)

    
    if os.path.isfile(args.pretrained_weights):

        print("args.pretrained_weights is", args.pretrained_weights)

        state_dict = torch.load(args.pretrained_weights, map_location="cpu")

        # # NOTE: check all keys in state_dict recursively
        # print("state_dict.keys() is", print_direct_subkeys(state_dict))

        # NOTE: load only the state_dicts
        state_dict = state_dict['state_dict']

        # NOTE: If you decide to use the ema_model or model weights:
        state_dict = {k.replace('ema_model.', ''): v for k, v in state_dict.items() if 'ema_model.' in k}
        # state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}



        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights found at {args.pretrained_weights}'
            # and loaded with msg: {}'.format(, msg)
              )
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # Define transform outside image processing pipeline
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # load and transform the image recursively from the folder
    if os.path.isdir(args.image_path):
        all_transformed_images_with_paths = process_images_recursively(args.image_path)
    else:
        print(f"The path {args.image_path} is not a valid directory.")
        exit()

    # Now iterate over each transformed image tensor and process it with your model
    for (img, image_path) in all_transformed_images_with_paths:
        # print("len(all_transformed_images) is", len(all_transformed_images))
        # print("img shape is", img.shape); exit()
        # print("img.device is", img.device); exit()

        # # tranform image
        # img = transform(img)

        # print("Image size : ", img.shape) [3, 480, 480]

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        # print(f"Image shape: {img.shape}") # [1, 3, 480, 480]

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # print("Feature map size : w, h", w_featmap, h_featmap) # 60, 60

        # NOTE: change this function to forward_features()
        # attentions = model.get_last_selfattention(img.to(device))

        attentions = model.forward_features(img.to(device), return_att=True, attn_stage=args.attn_stage)
        # print("attentions shape", attentions.shape) # [1, 1, 14400, 225] == [1, 1, 120*120, 15*15] for first stage
        # exit()

        nh = attentions.shape[1]  # number of heads

        for ref_index in range(0, args.ref_index):

            print("Processing ref_index:", ref_index)
            current_attentions = attentions[0, :, :, ref_index].reshape(nh, -1)

            # Optional thresholding
            if args.threshold is not None:
                val, idx = torch.sort(current_attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

            # Reshape for saving
            nh, ne = current_attentions.shape
            side = int(np.sqrt(ne))
            current_attentions = current_attentions.reshape(nh, side, side)

            # NOTE: drop first row and first column if necessary
            current_attentions = current_attentions[:, 1:, 1:]

            # Interpolate to the desired size
            current_attentions = nn.functional.interpolate(current_attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

            # Create output directory for current ref_index
            output_dir = os.path.join(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the original image
            torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), 
                                        os.path.join(output_dir, "img.png"))
            
            # Save attention maps and thresholded attention maps (if applicable)
            for j in range(nh):
                
                # NOTE: this is important!
                fname = os.path.join(output_dir, f"ref_index_{ref_index}_{os.path.basename(image_path)}")

                plt.imsave(fname=fname, arr=current_attentions[j], format='png')
                print(f"Attention map saved: {fname}")

                if args.threshold is not None:
                    mask_fname = os.path.join(output_dir, f"mask_th{args.threshold}_head{j}.png")
                    image = skimage.io.imread(os.path.join(output_dir, "img.png"))
                    display_instances(image, th_attn[j], fname=mask_fname, blur=False)
                    print(f"Thresholded attention map saved: {mask_fname}")