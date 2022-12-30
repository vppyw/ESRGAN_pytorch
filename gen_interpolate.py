import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

def main(args):
    hr_fnames = os.listdir(args.img_dir)
    sorted(hr_fnames)
    os.makedirs(args.save_dir, exist_ok=True)
    m0 = torch.load(args.model0, map_location=args.device)
    state_dict0 = m0.state_dict()
    keys0 = list(state_dict0.keys())
    m1 = torch.load(args.model1, map_location=args.device)
    state_dict1 = m1.state_dict()
    keys1 = list(state_dict1.keys())
    if not keys0 == keys1:
        print("The models do not have the save structure!!!")
        raise RuntimeError
    new_m = torch.load(args.model0, map_location=args.device)
    for hr_fname in tqdm(hr_fnames, ncols=50):
        img = torch.Tensor([])
        hr_img = torchvision.io.read_image(
                    os.path.join(args.img_dir, hr_fname),
                    mode=torchvision.io.ImageReadMode.RGB
                  ).float().to(args.device).div(255).unsqueeze(dim=0)
        if hr_img.size(-1) % args.scale_factor != 0:
            hr_img = hr_img[:,:,:,:-(hr_img.size(-1) % args.scale_factor)]
        if hr_img.size(-2) % args.scale_factor != 0:
            hr_img = hr_img[:,:,:-(hr_img.size(-2) % args.scale_factor),:]
        lr_img = F.interpolate(
                    hr_img,
                    scale_factor=1/args.scale_factor,
                    mode="bicubic"
                 ).clamp(min=0, max=1)
        with torch.no_grad():
            for p in tuple(val/4 for val in range(5)):
                state_dict = {}
                for key in keys0:
                    state_dict[key] = p * state_dict0[key] \
                                      + (1 - p) * state_dict1[key]
                new_m.load_state_dict(state_dict)
                rec_img = new_m(lr_img).clamp(min=0, max=1)
                img = torch.cat(
                        [img, x:=rec_img.mul(255).squeeze().to("cpu")],
                        dim=-1
                      )
        torchvision.io.write_png(
            img.to(torch.uint8),
            os.path.join(args.save_dir, f"inter_{hr_fname}")
        )
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model0", type=str, required=True)
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--scale_factor", type=int, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
