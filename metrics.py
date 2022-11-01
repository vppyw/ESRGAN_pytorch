import torch
import torchvision
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from argparse import ArgumentParser

def main(args):
    hr_img = (torchvision.io.read_image(args.hr_img).to(args.device) / 255.0).unsqueeze(dim=0)
    if args.load_model:
        model = torch.load(args.model).to(args.device)
        lr_img = (torchvision.io.read_image(args.lr_img).to(args.device) / 255.0).unsqueeze(dim=0)
        rec_img = model(lr_img)
    else:
        rec_img = (torchvision.io.read_image(args.lr_img).to(args.device) / 255.0).unsqueeze(dim=0)
    print(f"PSNR {PSNR(rec_img, hr_img)}")
    print(f"SSIM {SSIM(rec_img, hr_img)}")
    if args.do_save:
        rec_img = 255 * rec_img.squeeze()
        torchvision.io.write_png(rec_img.to(torch.uint8).to("cpu"), args.fname)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")
    parser.add_argument("--lr_img", type=str, required=True, help="low resolution image file")
    parser.add_argument("--hr_img", type=str, required=True, help="high resolution image file")
    parser.add_argument("--load_model", action="store_true", help="Load SR model")
    parser.add_argument("--model", type=str, help="SR model file name")
    parser.add_argument("--do_save", action="store_true", help="Save generated image")
    parser.add_argument("--fname", type=str, help="Save generated image file name")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
