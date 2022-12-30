import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM

from argparse import ArgumentParser
from tqdm import tqdm

import sr_models
import hr_dataset
import kd_utils

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, device: str, f_name: str):
    model = model.to("cpu")
    model.eval()
    torch.save(model, f_name)
    model = model.to(device)

def PSNRtrain(args, model, teacher_model, trainloader, validloader):
    model = model.to(args.device)
    teacher_model = teacher_model.to(args.device)
    teacher_model.eval()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.do_steplr:
        if args.steplr_mode == "batch":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                        step_size=args.steplr_step_size * len(trainloader),
                                                        gamma=args.steplr_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                        step_size=args.steplr_step_size,
                                                        gamma=args.steplr_gamma)

    l1_loss_fn = nn.L1Loss()
    kd_loss_fn = kd_utils.ATLoss(p=2)
    min_loss = float('inf')
    pbar = tqdm(range(args.num_epoch), ncols=50)
    for epoch in pbar:
        if args.do_train:
            train_loss = []
            train_l1_loss = []
            train_kd_loss = []
            train_psnr = []
            train_ssim = []
            model.train()
            optim.zero_grad()
            for idx, (lr_imgs, hr_imgs) in enumerate(trainloader):
                lr_imgs = lr_imgs.to(args.device)
                hr_imgs = hr_imgs.to(args.device)
                *s_hidden, rec_imgs = model(
                                        lr_imgs,
                                        return_hidden=True
                                     )
                with torch.no_grad():
                    *t_hidden, _ = teacher_model(
                                    lr_imgs,
                                    return_hidden=True
                                  )
                l1_loss = l1_loss_fn(rec_imgs, hr_imgs)
                kd_loss = sum(
                            kd_loss_fn(s, t) \
                            for s, t in zip(s_hidden, t_hidden)
                          )
                train_l1_loss.append(l1_loss.item())
                train_kd_loss.append(kd_loss.item())
                if epoch < args.kd_epoch:
                    loss = args.alpha * l1_loss + args.beta * kd_loss
                else:
                    loss = l1_loss
                train_loss.append(loss.item())
                loss.backward()
                if (idx + 1) % args.gradient_accumulation_steps == 0 or \
                    idx == len(trainloader) - 1:
                    optim.step()
                    if args.do_steplr:
                        scheduler.step()
                    optim.zero_grad()
                train_psnr.append(PSNR(rec_imgs, hr_imgs).item())
                train_ssim.append(SSIM(rec_imgs, hr_imgs).item())
            train_l1_loss = np.array(train_l1_loss).mean()
            train_kd_loss = np.array(train_kd_loss).mean()
            train_loss = np.array(train_loss).mean()
            train_psnr = np.array(train_psnr).mean()
            train_ssim= np.array(train_ssim).mean()
            pbar.write(f"|Epoch {epoch}|\n|train loss:{train_loss:.5e}|L1:{train_l1_loss:.5e}|KD_loss:{train_kd_loss:.5e}|PSNR:{train_psnr:.5}|SSIM:{train_ssim:.5}|")
            if args.do_save and not args.do_valid:
                save_model(model,
                           args.device,
                           args.model_file)
                pbar.write(f"Save {args.model_file} at {epoch}")

        if args.do_valid and epoch % args.valid_epoch == 0:
            with torch.no_grad():
                valid_loss = []
                valid_psnr = []
                valid_ssim = []
                model.eval()
                for lr_imgs, hr_imgs in validloader:
                    lr_imgs = lr_imgs.to(args.device)
                    hr_imgs = hr_imgs.to(args.device)
                    with torch.no_grad():
                        *s_hidden, rec_imgs = model(
                                                lr_imgs,
                                                return_hidden=True
                                             )
                        *t_hidden, _ = teacher_model(
                                        lr_imgs,
                                        return_hidden=True
                                      )
                    l1_loss = l1_loss_fn(rec_imgs, hr_imgs)
                    kd_loss = sum(
                                kd_loss_fn(s, t) \
                                for s, t in zip(s_hidden, t_hidden)
                              )
                    loss = args.alpha * l1_loss + args.beta * kd_loss
                    valid_loss.append(loss.item())
                    valid_psnr.append(PSNR(rec_imgs, hr_imgs).item())
                    valid_ssim.append(SSIM(rec_imgs, hr_imgs).item())
                valid_loss = np.array(valid_loss).mean()
                valid_psnr = np.array(valid_psnr).mean()
                valid_ssim = np.array(valid_ssim).mean()
                pbar.write(f"|Epoch {epoch}|valid loss:{valid_loss:.5e}|PSNR:{valid_psnr:.5}|SSIM:{valid_ssim:.5}|")
                if args.do_save and valid_loss < min_loss:
                    min_loss = valid_loss
                    save_model(model,
                               args.device,
                               args.model_file)
                    pbar.write(f"Save {args.model_file} at {epoch}")
    print("Finish PSNR training")
        
def main(args):
    same_seed(0)
    os.makedirs(args.cache, exist_ok=True)
    torch.hub.set_dir(args.cache)

    with open(args.config_file, "w") as f:
        f.write(f"{json.dumps(vars(args), indent=4)}\n")

    if args.do_save:
        args.model_file = os.path.join(args.save_dir,
                                       args.model_file)
    # Dataloader
    if args.do_train:
        trainset = hr_dataset.ImgDataset(dirname=args.train_dir[0],
                                         mode="train",
                                         scale_factor=1/args.scale_factor,
                                         crop_size=args.crop_size)
        for train_dir in args.train_dir[1:]:
            trainset += hr_dataset.ImgDataset(dirname=train_dir,
                                             mode="train",
                                             scale_factor=1/args.scale_factor,
                                             crop_size=args.crop_size)
        trainloader = DataLoader(trainset,
                                 batch_size=args.batch_size_per_gpu,
                                 shuffle=True,
                                 num_workers=args.num_workers)
    else:
        trainloader = None

    if args.do_valid:
        validset = hr_dataset.ImgDataset(dirname=args.valid_dir[0],
                                         mode="valid",
                                         scale_factor=1/args.scale_factor,
                                         crop_size=args.crop_size)
        for valid_dir in args.valid_dir[1:]:
            validset += hr_dataset.ImgDataset(dirname=valid_dir,
                                              mode="valid",
                                              scale_factor=1/args.scale_factor,
                                              crop_size=args.crop_size)
        validloader = DataLoader(validset,
                                 batch_size=args.batch_size_per_gpu,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    else:
        validloader = None
    # Models and Train
    if args.model_type == "psnr":
        if args.load_model != None:
            model = torch.load(args.load_model, map_location="cpu")
        else:
            model = sr_models.SRResNet(scale_factor=args.scale_factor,
                                       upscale_mode=args.upscale_mode,
                                       channels=args.channels)
        teacher_model = torch.load(args.load_teacher, map_location="cpu")
        PSNRtrain(args, model, teacher_model, trainloader, validloader)
    else:
        raise NotImplementedError

def parse_args():
    parser = ArgumentParser()
    # Computation Device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)

    # Environment
    parser.add_argument("--cache", type=str, default="./cache")
    parser.add_argument("--train_dir", type=str, default=None, nargs="*")
    parser.add_argument("--valid_dir", type=str, default=None, nargs="*")
    parser.add_argument("--test_dir", type=str, default=None, nargs="*")
    
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--valid_epoch", type=int, default=1)

    # Load model
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--load_teacher", type=str, required=True)

    # Save model
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--config_file", type=str, default="config.json")
    parser.add_argument("--save_dir", type=str, default="ckpt/")
    parser.add_argument("--model_file", type=str, default="model.pt")

    # Hyperparameters
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size_per_gpu", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=512)
    parser.add_argument("--kd_epoch", type=int, default=0)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--upscale_mode", type=str, default="nearest")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--growth_rate", type=int, default=32)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--do_steplr", action="store_true")
    parser.add_argument("--steplr_step_size", type=int)
    parser.add_argument("--steplr_gamma", type=float, default=0.5)
    parser.add_argument("--steplr_mode", type=str, default="batch")
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
