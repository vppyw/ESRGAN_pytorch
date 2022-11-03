import os
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

def PSNRtrain(args, model, trainloader, validloader):
    model = model.to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    min_loss = float('inf')
    pbar = tqdm(range(args.num_epoch), ncols=50)
    for epoch in pbar:
        if args.do_train:
            train_loss = []
            train_psnr = []
            train_ssim = []
            model.train()
            optim.zero_grad()
            for idx, (lr_imgs, hr_imgs) in enumerate(trainloader):
                lr_imgs = lr_imgs.to(args.device)
                hr_imgs = hr_imgs.to(args.device)
                rec_imgs = model(lr_imgs)
                loss = loss_fn(rec_imgs, hr_imgs)    
                train_loss.append(loss.item())
                loss.backward()
                if (idx + 1) % args.gradient_accumulation_steps == 0 or \
                    idx == len(trainloader) - 1:
                    optim.step()
                    optim.zero_grad()
                train_psnr.append(PSNR(rec_imgs, hr_imgs).item())
                train_ssim.append(SSIM(rec_imgs, hr_imgs).item())
            train_loss = np.array(train_loss).mean()
            train_psnr = np.array(train_psnr).mean()
            train_ssim= np.array(train_ssim).mean()
            pbar.write(f"|Epoch {epoch}|train loss:{train_loss:.5e}|PSNR:{train_psnr:.5}|SSIM:{train_ssim:.5}|")
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
                    rec_imgs = model(lr_imgs)
                    loss = loss_fn(rec_imgs, hr_imgs)    
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

def GANtrain(args, gene, disc, extr, trainloader, validloader):
    # Models
    gene = gene.to(args.device)
    disc = disc.to(args.device)
    extr = extr.to(args.device)
    extr.eval()

    # Loss functions
    l1_loss_fn = nn.L1Loss()
    gan_loss_fn = nn.BCEWithLogitsLoss()
    perc_loss_fn = nn.MSELoss()
    min_g_loss = float('inf')

    # Criterion
    gene_optim = torch.optim.Adam(gene.parameters(), lr=1e-4)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=1e-4)

    pbar = tqdm(range(args.num_epoch), ncols=50)
    for epoch in pbar:
        # Desciminator Training
        if args.do_train:
            train_gan_d_loss = []
            disc.train()
            gene.eval()
            disc_optim.zero_grad()
            for idx, (lr_imgs, hr_imgs) in enumerate(trainloader):
                lr_imgs = lr_imgs.to(args.device)
                hr_imgs = hr_imgs.to(args.device)
                gene_imgs = gene(lr_imgs)
                hr_logits = disc(hr_imgs)
                gene_logits = disc(hr_imgs)
                gan_d_loss = gan_loss_fn(hr_imgs - gene_imgs.mean(),
                                         torch.ones_like(hr_imgs).to(args.device))\
                             + gan_loss_fn(gene_imgs - hr_imgs.mean(),
                                           torch.zeros_like(gene_imgs).to(args.device))
                gan_d_loss.backward()

                if (idx + 1) % args.gradient_accumulation_steps == 0 or \
                   idx == len(trainloader) - 1:
                    disc_optim.step()
                    disc_optim.zero_grad()
                train_gan_d_loss.append(gan_d_loss.item())
            train_gan_d_loss = np.array(train_gan_d_loss).mean()
            pbar.write(f"|Epoch:{epoch}|Discriminator|train:{train_gan_d_loss}|")

        # Desciminator Validation
        if args.do_valid and epoch % args.valid_epoch == 0:
            valid_gan_d_loss = []
            disc.eval()
            gene.eval()
            with torch.no_grad():
                for lr_imgs, hr_imgs in validloader:
                    lr_imgs = lr_imgs.to(args.device)
                    hr_imgs = hr_imgs.to(args.device)
                    hr_logits = disc(hr_imgs)
                    gene_imgs = gene(lr_imgs)
                    gene_logits = disc(gene_imgs)
                    gan_d_loss = gan_loss_fn(hr_imgs - gene_imgs.mean(),
                                             torch.ones_like(hr_imgs).to(args.device))\
                               + gan_loss_fn(gene_imgs - hr_imgs.mean(),
                                             torch.zeros_like(gene_imgs).to(args.device))
                    valid_gan_d_loss.append(gan_d_loss.item())
                valid_gan_d_loss = np.array(valid_gan_d_loss).mean()
                pbar.write(f"|Epoch:{epoch}|Discriminator|valid:{valid_gan_d_loss}|")
        if args.do_save:
            save_model(disc,
                       args.device,
                       args.disc_file)
        # Generator Training
        if args.do_train:
            train_l1_loss = []
            train_gan_g_loss = []
            train_perc_loss = []
            train_psnr = []
            train_ssim = []
            disc.eval()
            gene.train()
            gene_optim.zero_grad()
            for idx, (lr_imgs, hr_imgs) in enumerate(trainloader):
                lr_imgs = lr_imgs.to(args.device)
                hr_imgs = hr_imgs.to(args.device)
                hr_logits = disc(hr_imgs)
                hr_h, hr_l = extr(hr_imgs)
                gene_imgs = gene(lr_imgs)
                gene_logits = disc(gene_imgs)
                gene_h, gene_l = extr(gene_imgs)
                l1_loss = l1_loss_fn(gene_imgs, hr_imgs)
                perc_loss = perc_loss_fn(gene_h, hr_h) + perc_loss_fn(gene_l, hr_l)
                gan_loss = gan_loss_fn(hr_logits - gene_logits.mean(),
                                       torch.zeros_like(hr_logits))\
                         + gan_loss_fn(gene_logits - hr_logits.mean(),
                                       torch.ones_like(gene_logits))
                loss = l1_loss + perc_loss + gan_loss
                train_l1_loss.append(l1_loss.item())
                train_perc_loss.append(perc_loss.item())
                train_gan_g_loss.append(gan_loss.item())
                loss.backward()
                if (idx + 1) % args.gradient_accumulation_steps  == 0 or \
                   idx == len(trainloader) - 1:
                    gene_optim.step()
                    gene_optim.zero_grad()
                train_psnr.append(PSNR(gene_imgs, hr_imgs).item())
                train_ssim.append(SSIM(gene_imgs, hr_imgs).item())
            train_l1_loss = np.array(train_l1_loss).mean()
            train_perc_loss = np.array(train_perc_loss).mean()
            train_gan_g_loss = np.array(train_gan_g_loss).mean()
            train_psnr = np.array(train_psnr).mean()
            train_ssim = np.array(train_ssim).mean()
            pbar.write(f"|Epoch:{epoch}|Generator|")
            pbar.write(f"|Train|L1:{train_l1_loss:.5e}|\
Perceptual:{train_perc_loss:.5e}|\
GAN:{train_gan_g_loss:.5e}|")
            pbar.write(f"|PSNR:{train_psnr}|SSIM:{train_ssim}|")

        # Generator Validation
        if args.do_valid and epoch % args.valid_epoch == 0:
            valid_l1_loss = []
            valid_gan_g_loss = []
            valid_perc_loss = []
            valid_psnr = []
            valid_ssim = []
            disc.eval()
            gene.eval()
            with torch.no_grad():
                for lr_imgs, hr_imgs in validloader:
                    lr_imgs = lr_imgs.to(args.device)
                    hr_imgs = hr_imgs.to(args.device)
                    hr_logits = disc(hr_imgs)
                    hr_h, hr_l = extr(hr_imgs)
                    gene_imgs = gene(lr_imgs)
                    gene_logits = disc(gene_imgs)
                    gene_h, gene_l = extr(gene_imgs)
                    l1_loss = l1_loss_fn(gene_imgs, hr_imgs)
                    perc_loss = perc_loss_fn(gene_h, hr_h) + perc_loss_fn(gene_l, hr_l)
                    gan_loss = gan_loss_fn(hr_logits - gene_logits.mean(),
                                           torch.zeros_like(hr_logits))\
                             + gan_loss_fn(gene_logits - hr_logits.mean(),
                                           torch.ones_like(gene_logits))
                    valid_l1_loss.append(l1_loss.item())
                    valid_perc_loss.append(perc_loss.item())
                    valid_gan_g_loss.append(gan_loss.item())
                    valid_psnr.append(PSNR(gene_imgs, hr_imgs).item())
                    valid_ssim.append(SSIM(gene_imgs, hr_imgs).item())
                valid_l1_loss = np.array(valid_l1_loss).mean()
                valid_perc_loss = np.array(valid_perc_loss).mean()
                valid_gan_g_loss = np.array(valid_gan_g_loss).mean()
                valid_psnr = np.array(valid_psnr).mean()
                valid_ssim = np.array(valid_ssim).mean()
                pbar.write(f"|Epoch:{epoch}|Generator|")
                pbar.write(f"|Valid|L1:{valid_l1_loss:.5e}|\
Perceptual:{valid_perc_loss:.5e}|\
GAN:{valid_gan_g_loss:.5e}|")
                pbar.write(f"|PSNR:{valid_psnr}|SSIM:{valid_ssim}|")
        if args.do_save:
            save_model(gene,
                       args.device,
                       args.gene_file)
        
def main(args):
    same_seed(0)
    # Dataloader
    if args.do_train:
        trainset = hr_dataset.ImgDataset(dirname=args.train_dir[0],
                                         mode="train",
                                         scale_factor=1/args.scale_factor,
                                         crop_size=args.crop_size)
        for train_dir in args.train_dir[1:]:
            trainset = hr_dataset.ImgDataset(dirname=train_dir,
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
            model = torch.load(args.load_model)
        else:
            model = sr_models.SRResNet(scale_factor=args.scale_factor,
                                       upscale_mode=args.upscale_mode)
        PSNRtrain(args, model, trainloader, validloader)
    elif args.model_type == "gan":
        if args.load_gene != None:
            gene = torch.load(args.load_gene)
        else:
            gene = sr_models.SRResNet(scale_factor=args.scale_factor,
                                      upscale_mode=args.upscale_mode)
        if args.load_disc != None:
            disc = torch.load(args.load_disc)
        else:
            disc = sr_models.Discriminator()
        if args.extr_type == "vgg19":
            extr = sr_models.VGG19()
        else:
            raise NotImplementedError
        GANtrain(args, gene, disc, extr, trainloader, validloader)
    else:
        raise NotImplementedError

def parse_args():
    parser = ArgumentParser()
    # Computation Device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)

    # Environment
    parser.add_argument("--train_dir", type=str, default=None, nargs="*")
    parser.add_argument("--valid_dir", type=str, default=None, nargs="*")
    parser.add_argument("--test_dir", type=str, default=None, nargs="*")
    
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--valid_epoch", type=int, default=1)

    # Load model
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--load_gene", type=str, default=None)
    parser.add_argument("--load_disc", type=str, default=None)
    parser.add_argument("--extr_type", type=str, default="vgg19")

    # Save model
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--model_file", type=str, default="model.pt")
    parser.add_argument("--gene_file", type=str, default="gene.pt")
    parser.add_argument("--disc_file", type=str, default="disc.pt")

    # Hyperparameters
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size_per_gpu", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=512)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--upscale_mode", type=str, default="nearest")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
