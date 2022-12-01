import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class RGBShuffle(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[torch.randperm(3),:,:]

class ImgDataset(Dataset):
    def __init__(self, dirname: str, mode: str="train", scale_factor=0.25, crop_size=256):
        super().__init__()
        self.fnames = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        self.mode = mode
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.resize_tfm = transforms.Resize(self.crop_size)
        if self.mode == "train":
            self.tfm = transforms.Compose([
                            transforms.RandomCrop((self.crop_size, self.crop_size)),
                            transforms.RandomChoice(
                                (transforms.ColorJitter(),
                                 RGBShuffle()),
                                p=(0.4,
                                   0.2)
                            ),
                            transforms.RandomHorizontalFlip(p=0.5), 
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomChoice(
                                (transforms.RandomRotation((90, 90)),
                                 transforms.RandomRotation((-90, -90))),
                                p=(0.25, 0.25),
                            )
                        ])
            self.lr_tfm = transforms.Compose([
                            transforms.RandomChoice(
                                (
                                    transforms.GaussianBlur(kernel_size=3),
                                    transforms.GaussianBlur(kernel_size=5),
                                ),
                                p=(0.2, 0.1)
                            )
                          ])
        elif self.mode == "valid":
            self.tfm = transforms.Compose([
                                  transforms.CenterCrop((self.crop_size, self.crop_size)),
                              ])

    def __getitem__(self, idx):
        """
        mode: train
            return LR, HR
        mode: valid
            return LR, HR
        mode: test
            return LR, LR
        """
        if self.mode in {"train", "valid"}:
            fname = self.fnames[idx]
            img = torchvision.io.read_image(fname,
                                            mode=torchvision.io.ImageReadMode.RGB).float() / 255
            if img.size(1) < self.crop_size or \
               img.size(2) < self.crop_size:
                img = self.resize_tfm(img)
            hr_img = self.tfm(img)
            lr_img = F.interpolate(hr_img.unsqueeze(dim=0),
                                   mode="bicubic",
                                   scale_factor=self.scale_factor)\
                                   .squeeze()
            lr_img = lr_img.clamp(min=0, max=1)
            return lr_img, hr_img
        elif self.mode == "test":
            fname = self.fnames[idx]
            lr_img = torchvision.io.read_image(fname,
                                               mode=torchvision.io.ImageReadMode.RGB).float() / 255
            lr_img = lr_img.clamp(min=0, max=1)
            return lr_img, lr_img
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.fnames)
