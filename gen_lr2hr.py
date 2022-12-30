import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import cv2
import numpy as np
from tqdm import tqdm

def torch_rgb2ycrcb(img):
    device = img.device
    out = img.squeeze().permute(1, 2, 0).cpu().numpy()
    out = cv2.cvtColor(out, cv2.COLOR_RGB2YCR_CB)
    out = torch.from_numpy(out).permute(2, 0, 1)
    return out.to(device)

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    with torch.no_grad():
        model = torch.load(args.model).to(args.device)
        lr_fnames = os.listdir(args.img_dir)
        for idx, lr_fname in enumerate(tqdm(lr_fnames, ncols=50)):
            lr_img = torchvision.io.read_image(
                        os.path.join(args.img_dir, lr_fname),
                        mode=torchvision.io.ImageReadMode.RGB
                     ).div(255).float().to(args.device).unsqueeze(dim=0)
            rec_img = model(lr_img).clamp(min=0, max=1)
            rec_img = 255 * rec_img.squeeze()
            torchvision.io.write_png(
                rec_img.to(torch.uint8).to("cpu"),
                os.path.join(args.result_dir, f"result_{lr_fname}")
            )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")
    parser.add_argument("--model", type=str, help="SR model file name", required=True)
    parser.add_argument("--img_dir", type=str, help="[batch mode] high resolution images directory", required=True)
    parser.add_argument("--result_dir", type=str, help="[batch mode] Save generated image file name", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
