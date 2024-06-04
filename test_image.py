import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import VitMatte2

from model.model import HumanSegment, HumanMatting
import utils
import inference

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--image-path', type=str, required=True)
parser.add_argument('--result-dir', type=str, default='./results')
parser.add_argument('--maskedimage-path', type=str, default = './masked_images/')
parser.add_argument('--pretrained-weight', type=str, default= './pretrained/SGHM-ResNet50.pth')

args = parser.parse_args()

if not os.path.exists(args.pretrained_weight):
    print('Cannot find the pretrained model: {0}'.format(args.pretrained_weight))
    exit()

# --------------- Main ---------------
# Load Model
model = HumanMatting(backbone='resnet50')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model = nn.DataParallel(model).cuda().eval()
    model.load_state_dict(torch.load(args.pretrained_weight))
else:
    state_dict = torch.load(args.pretrained_weight, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
model.eval()
print("Load checkpoint successfully ...")

# Load Image
image_path = args.image_path
output_path = args.maskedimage_path

if not os.path.exists(image_path):
    print('Cannot find the image: {0}'.format(image_path))
    exit()

print("Processing image:", image_path)
with Image.open(image_path) as img:
    img = img.convert("RGB")

# Inference
pred_alpha, pred_mask = inference.single_inference(model, img, device=device)

# Save Results
output_dir = args.result_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_name = os.path.splitext(os.path.basename(image_path))[0]
save_path = os.path.join(output_dir, f"{image_name}_trimap.png")
Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)

print("Trimap saved:", save_path)

VitMatte.apply_trimap_to_image(image_path, save_path, output_path)

