import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImgDataset(Dataset):
    def __init__(self, dirname: str, mode: str="train", scale_factor=0.25):
        super().__init__()
        self.fnames = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        self.mode = mode
        self.scale_factor = scale_factor
        if self.mode == "train":
            self.transforms = transforms.Compose([
                                  transforms.AutoAugment(),
                                  transforms.RandomRotation((-90, 90)),
                                  transforms.RandomCrop((128, 128)),
                              ])
        elif self.mode == "valid":
            self.transforms = transforms.Compose([
                                  transforms.RandomCrop((128, 128)),
                              ])

    def __getitem__(self, idx):
        """
        mode: train
            return LR, HR
        mode: valid
            return LR, LR
        mode: test
            return LR, None
        """
        if self.mode in {"train", "valid"}:
            fname = self.fnames[idx]
            img = torchvision.io.read_image(fname)
            hr_img = self.transforms(img).float()
            lr_img = F.interpolate(hr_img.unsqueeze(dim=0),
                                   scale_factor=self.scale_factor)\
                                   .squeeze()
            return lr_img, hr_img
        elif self.mode == "test":
            fname = self.fnames[idx]
            lr_img = torchvision.io.read_image(fname).float()
            return lr_img.long(), None
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.fnames)
