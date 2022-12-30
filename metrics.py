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
    if args.do_save:
        os.makedirs(args.result_dir, exist_ok=True)
    if args.mode == "single":
        with torch.no_grad():
            hr_img = (torchvision.io.read_image(args.hr_img)\
                        .float().to(args.device)).unsqueeze(dim=0) / 255
            if args.load_model:
                model = torch.load(args.model).to(args.device)
                lr_img = (torchvision.io.read_image(args.lr_img).float().to(args.device)).unsqueeze(dim=0)
                rec_img = model(lr_img)
            else:
                rec_img = (torchvision.io.read_image(args.lr_img).to(args.device)).unsqueeze(dim=0)
            if args.color_space == "ycrcb":
                hr_img_y = torch_rgb2ycrcb(hr_img).unsqueeze(dim=0)
                rec_img_y = torch_rgb2ycrcb(rec_img).unsqueeze(dim=0)
            else:
                hr_img_y = hr_img
                rec_img_y = rec_img
            print(f"PSNR {PSNR(rec_img_y, hr_img_y)}")
            print(f"SSIM {SSIM(rec_img_y, hr_img_y)}")
            if args.do_save:
                rec_img = 255 * rec_img.squeeze()
                torchvision.io.write_png(rec_img.to(torch.uint8).to("cpu"), args.fname)
    elif args.mode == "batch":
        with torch.no_grad():
            if args.load_model:
                model = torch.load(args.model).to(args.device)
            else:
                print("Load model in batch mode")
                exit()
            psnr = []
            ssim = []
            hr_fnames = os.listdir(args.img_dir)
            for idx, hr_fname in enumerate(tqdm(hr_fnames, ncols=50)):
                hr_img = torchvision.io.read_image(
                            os.path.join(args.img_dir, hr_fname),
                            mode=torchvision.io.ImageReadMode.RGB
                         ).div(255).float().to(args.device).unsqueeze(dim=0)

                if hr_img.size(-1) % args.scale_factor != 0:
                    hr_img = hr_img[:,:,:,:-(hr_img.size(-1) % args.scale_factor)]
                if hr_img.size(-2) % args.scale_factor != 0:
                    hr_img = hr_img[:,:,:-(hr_img.size(-2) % args.scale_factor),:]
                lr_img = F.interpolate(hr_img,
                                       scale_factor=1/args.scale_factor,
                                       mode="bicubic").clamp(min=0, max=1)
                rec_img = model(lr_img).clamp(min=0, max=1)
                if args.color_space == "ycrcb":
                    hr_img_y = torch_rgb2ycrcb(hr_img).unsqueeze(dim=0)
                    rec_img_y = torch_rgb2ycrcb(rec_img).unsqueeze(dim=0)
                else:
                    hr_img_y = hr_img
                    rec_img_y = rec_img 
                    psnr.append(PSNR(rec_img_y, hr_img_y).item())
                    ssim.append(SSIM(rec_img_y, hr_img_y).item())
                if args.do_save:
                    rec_img = 255 * rec_img.squeeze()
                    torchvision.io.write_png(
                        rec_img.to(torch.uint8).to("cpu"),
                        os.path.join(args.result_dir, f"result_{hr_fname}")
                    )
            print(f"PSNR: {np.array(psnr).mean()}")
            print(f"SSIM: {np.array(ssim).mean()}")
    else:
        raise NotImplementedError

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")
    parser.add_argument("--load_model", action="store_true", help="Load SR model")
    parser.add_argument("--model", type=str, help="SR model file name")
    parser.add_argument("--scale_factor", type=int)
    parser.add_argument("--mode", type=str, required=True, help="single image mode or batch mode")
    parser.add_argument("--lr_img", type=str, help="[single mode] low resolution image file")
    parser.add_argument("--hr_img", type=str, help="[single mode] high resolution image file")
    parser.add_argument("--img_dir", type=str, help="[batch mode] high resolution images directory")
    parser.add_argument("--do_save", action="store_true", help="Save generated image")
    parser.add_argument("--fname", type=str, help="[single mode] Save generated image file name")
    parser.add_argument("--result_dir", type=str, help="[batch mode] Save generated image file name")
    parser.add_argument("--color_space", type=str, default="rgb")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
