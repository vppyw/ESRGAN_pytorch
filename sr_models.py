import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, vgg19, VGG19_Weights

class DenseBlock(nn.Module):
    def __init__(self, channels: int, growth_rate: int, dropout: float=0.0, beta: float=0.2):
        """
        channels: input channels, output channels
        growth_rate: the number of channels that increase in each layer of conv
        dropout: dropout
        beta: out = beta * conv(x) + x
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels + growth_rate * 0, growth_rate,
                                kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate * 1, growth_rate,
                                kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels + growth_rate * 2, growth_rate,
                                kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels + growth_rate * 3, growth_rate,
                                kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels + growth_rate * 4, channels,
                                kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.dropout = nn.Dropout(p=dropout)
        self.beta = beta
        self.low_init_weights()

    def low_init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, H, W)
        out: (N, C, H, W)
        """
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat((x, out1), dim=1)))
        out3 = self.lrelu(self.conv3(torch.cat((x, out1, out2), dim=1)))
        out4 = self.lrelu(self.conv4(torch.cat((x, out1, out2, out3), dim=1)))
        out5 = self.dropout(self.conv5(torch.cat((x, out1, out2, out3, out4), dim=1)))
        out = torch.add(x, torch.mul(out5, self.beta))
        return out

class RRDB(nn.Module):
    def __init__(self, channels, growth_rate, dropout: float=0.0, beta: float=0.2):
        """
        channels: input channels, output channels
        growth_rate: the number of channels that increase in each layer of conv
        dropout: dropout
        beta(for DenseBlock): out = beta * conv(x) + x
        """
        super().__init__()
        self.db1 = DenseBlock(channels, growth_rate, dropout, beta)
        self.db2 = DenseBlock(channels, growth_rate, dropout, beta)
        self.db3 = DenseBlock(channels, growth_rate, dropout, beta)
        self.beta = beta

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, H, W)
        out: (N, C, H, W)
        """
        out = self.db1(x)
        out = self.db2(torch.add(x, torch.mul(out, self.beta)))
        out = self.db3(torch.add(x, torch.mul(out, self.beta)))
        out = torch.add(x, torch.mul(out, self.beta))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, bn: bool=False):
        if bn:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BachNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BachNorm2d(in_channels),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, channels,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
            )
        self.low_init_weights()

    def low_init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, H, W)
        out: (N, C, H, W)
        """
        return torch.add(x, self.net(x))
        
class SRResNet(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=3,
                 channels: int=64,
                 growth_rate: int=32,
                 dropout: float=0.0,
                 beta: float=0.2,
                 num_blocks: int=23,
                 block_type: str="rrdb",
                 scale_factor: int=4,
                 upscale_mode: str="nearest"):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels,
                               kernel_size=3, stride=1, padding=1)
        if block_type == "rrdb":
            blocks = (RRDB(channels, growth_rate, dropout, beta) for _ in range(num_blocks))
            self.blocks = nn.Sequential(*blocks)
        elif block_type == "rb":
            blocks = (ResidualBlock(channels, channels, bn=False) for _ in range(num_blocks))
            self.blocks = nn.Sequential(*blocks)
        elif block_type == "rb_bn":
            blocks = (ResidualBlock(channels, channels, bn=True) for _ in range(num_blocks))
            self.blocks = nn.Sequential(*blocks)
        else:
            raise NotImplementedError

        if scale_factor == 2:
            self.upsample = nn.Sequential(
                                nn.Upsample(scale_factor=2, mode=upscale_mode),    
                                nn.Conv2d(channels, channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.1, True),
                            )
        elif scale_factor == 4:
            self.upsample = nn.Sequential(
                                nn.Upsample(scale_factor=2, mode=upscale_mode),    
                                nn.Conv2d(channels, channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.1, True),
                                nn.Conv2d(channels, channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.1, True),
                                nn.Upsample(scale_factor=2, mode=upscale_mode),    
                                nn.Conv2d(channels, channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.1, True),
                                nn.Conv2d(channels, channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(0.1, True),
                            )
        else:
            raise NotImplementedError

        self.conv_out = nn.Sequential(
                            nn.Conv2d(channels, channels,
                                      kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(0.1, True),
                            nn.Conv2d(channels, out_channels,
                                      kernel_size=3, stride=1, padding=1),
                        )
        self.low_init_weights()

    def forward(self, x, return_hidden=False):
        """
        x: (N, C_in, H, W)
        out: (N, C_out, scale_factor * H, scale_factor * W)
        """
        if return_hidden:
            pass
        else:
            out = self.conv_in(x)
            out = self.blocks(out)
            out = self.upsample(out)
            out = self.conv_out(out)
            return out

    def low_init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, model_type: str="vgg19"):
        super().__init__()
        if model_type == "vgg19":
            self.cnn = torchvision.models.vgg19(weights=None)
        elif model_type == "resnet34":
            self.cnn = torchvision.models.resnet34(weights=None)
        else:
            raise NotImplementedError

        self.fc = nn.Sequential(
                     nn.Linear(1000, 128),
                     nn.LeakyReLU(0.1, True),
                     nn.Linear(128, 1),
                  )

    def forward(self, x):
        """
        x: (N, C, H, W)
        out: (N, 1)
        """
        out = self.cnn(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # pre_vgg19 = vgg19(weights=None)
        pre_vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
        self.h_features = pre_vgg19.features[:3]
        self.l_features = pre_vgg19.features[3:22]

    def forward(self, x):
        """
        x: (N, 3, 128, 128)
        h_out: (N, 64, 128, 128)
        l_out: (N, 512, 16, 16)
        """
        h_out = self.h_features(x)
        l_out = self.l_features(h_out)
        return h_out, l_out
