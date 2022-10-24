import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataloader
from argparse import ArgumentParser

from tqdm import tqdm

import model
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

def PNSRtrain(args, model, trainloader, validloader):
    model = model.to(args.device)
    lr = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn= nn.L1Loss()
    pbar = tqdm(range(args.pnsr_epoch), ncols=50)
    for epoch in pbar:
        train_loss = torch.Tensor([])
        valid_loss = torch.Tensor([])
        model.train()
        for lr_imgs, hr_imgs in range(trainloader):
            lr_imgs = lr_imgs.to(args.device)
            hr_imgs = hr_imgs.to(args.device)
            rec_imgs = model(lr_imgs)
            loss = loss_fn(rec_imgs, hr_imgs)    
            train_loss = torch.cat((train_loss,
                                    torch.Tensor([loss.item()])))
            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            model.eval()
            for lr_imgs, hr_imgs in range(validloader):
                lr_imgs = lr_imgs.to(args.device)
                hr_imgs = hr_imgs.to(args.device)
                rec_imgs = model(lr_imgs)
                loss = loss_fn(rec_imgs, hr_imgs)    
                valid_loss = torch.cat((valid_loss,
                                        torch.Tensor([loss.item()])))
        pbar.write(f"|Epoch {epoch}|\
train loss:{train_loss.mean().item()}|\
valid loss: {valid_loss.mean().item()}|")
        save_model(model,
                   args.device,
                   os.path.join(args.ckpt_dir, "psnr"+args.ckpt_file))
    print("Finish PSNR training")

def GANtrain(args, gene, disc, extr, trainloader, validloader):
    # Models
    gene = gene.to(args.device)
    disc = disc.to(args.device)
    extr = extr.to(args.device)
    extr.eval()

    # Loss functions
    l1_loss_fn = nn.L1Loss()
    gan_loss_fn = nn.BCEwithLogitsLoss()
    perc_loss_fn = nn.MSELoss()

    # Criterion
    gene_optim = torch.optim.Adam(gene.parameters(), lr=1e-4)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=1e-4)

    pbar = tqdm(range(args.gan_epoch), ncols=50)
    for epoch in pbar:
        train_gan_d_loss = torch.Tensor([])
        valid_gan_d_loss = torch.Tensor([])
        # Desciminator Training
        disc.train()
        gene.eval()
        for lr_imgs, hr_imgs in trainloader:
            lr_imgs = lr_imgs.to(args.device)
            hr_imgs = hr_imgs.to(args.device)
            gene_imgs = gene(lr_imgs)
            hr_logits = disc(hr_imgs)
            gene_logits = disc(hr_imgs)
            gan_d_loss = gan_loss_fn(hr_imgs - gene_imgs.mean(),
                                     torch.ones_like(hr_imgs).to(args.device))\
                       + gna_loss_fn(gene_imgs - hr_imgs.mean(),
                                     torch.zeros_like(gene_imgs).to(args.device))
            train_gan_d_loss = torch.cat((train_gan_d_loss,
                                          torch.Tensor([gan_d_loss.item()])))
            disc_optim.zero_grad()
            gan_d_loss.backward()
            disc_optim.step()

        # Desciminator Validation
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
                           + gna_loss_fn(gene_imgs - hr_imgs.mean(),
                                         torch.zeros_like(gene_imgs).to(args.device))
                valid_gan_d_loss = torch.cat((valid_gan_d_loss,
                                              torch.Tensor([gan_d_loss.item()])))
            
        pbar.write(f"|Epoch:{epoch}|Discriminator|\
train:{train_gan_d_loss.mean()}|\
valid:{valid_gan_d_loss.mean()}|\n")

        save_model(disc,
                   args.device,
                   os.path.join(args.ckpt_dir, "disc"+args.ckpt_file))

        train_l1_loss = torch.Tensor([])
        train_gan_g_loss = torch.Tensor([])
        train_perc_loss = torch.Tensor([])
        valid_l1_loss = torch.Tensor([])
        valid_gan_g_loss = torch.Tensor([])
        valid_perc_loss = torch.Tensor([])

        # Generator Training
        disc.eval()
        gene.train()
        for lr_imgs, hr_imgs in trainloader:
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
            train_l1_loss = torch.cat((train_l1_loss,
                                       torch.Tensor([l1_loss.item()])))
            train_perc_loss = torch.cat((train_perc_loss,
                                         torch.Tensor([perc_loss.item()])))
            train_gan_g_loss = torch.cat((train_gan_g_loss,
                                          torch.Tensor([gan_loss.item()])))
            gene_optim.zero_grad()
            loss.backward()
            gene_optim.step()

        # Generator Validation
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
                valid_l1_loss = torch.cat((valid_l1_loss,
                                           torch.Tensor([l1_loss.item()])))
                valid_perc_loss = torch.cat((valid_perc_loss,
                                             torch.Tensor([perc_loss.item()])))
                valid_gan_g_loss = torch.cat((valid_gan_g_loss,
                                              torch.Tensor([gan_loss.item()])))
        pbar.write("|Epoch:{epoch}|Generator|\n")
        pbar.write("Train: L1 {train_l1_loss.mean()}\
Perceptual {train_perc_loss.mean()}\
GAN {train_gan_g_loss.mean()}|\n")
        pbar.write("Valid: L1 {valid_l1_loss.mean()}\
Perceptual {valid_perc_loss.mean()}\
GAN {valid_gan_g_loss.mean()}|\n")

        save_model(gene,
                   args.device,
                   os.path.join(args.ckpt_dir, "gene"+args.ckpt_file))
        
def main(args):
    same_seed(0)
    trainset = hr_dataset(fname=args.train_dir, mode="train")
    validset = hr_dataset(fname=args.valid_dir, mode="valid")
    trainloader = Dataloader(trainset, batchsize=args.batch_size, shuffle=True)
    validloader = Dataloader(validset, batchsize=args.batch_size, shuffle=false)
    # TODO:Train PNSR model
    # TODO:Train GAN

def parse_args():
    parser = ArgumentParser()
    # Environment
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        require=True,
    )
    parser.add_argument(
        "--valid_dir",
        type=str,
        require=True,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpt/"
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="_ckpt.pt"
    )
    # Hyperparameters

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--pnsr_epoch",
        type=int,
        default=512,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
