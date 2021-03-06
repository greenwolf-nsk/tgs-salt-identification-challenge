from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.functional import upsample
from pretrainedmodels import se_resnext50_32x4d
from torchsummary import summary

from lib.timers import Timer


class SeResNextUnet(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, pretrained=False):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
        """
        super().__init__()

        self.pool = nn.AvgPool2d(2, 2)

        if pretrained:
            encoder = se_resnext50_32x4d(pretrained='imagenet')
        else:
            encoder = se_resnext50_32x4d(pretrained=None)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(3, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.center = nn.Sequential(
            ConvRelu(2048, 1024),
            ConvRelu(1024, 512),
        )
        self.dec5 = DecoderBlockSCSE(2560, 256, 256)
        self.dec4 = DecoderBlockSCSE(1280, 128, 128)
        self.dec3 = DecoderBlockSCSE(640, 64, 64)
        self.dec2 = DecoderBlockSCSE(320, 32, 32)
        self.dec1 = ConvRelu(96, 64)
        self.final_do = nn.Dropout(p=0.25)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.logit = nn.Sequential(
            nn.Conv2d(544, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2592, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv5_pool = nn.functional.avg_pool2d(conv5, kernel_size=8).view(x.size(0), -1)
        classification = self.fc(conv5_pool)

        center = self.center(conv5)  # print(center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        hypercolumn = torch.cat([
            dec1,
            dec2,
            upsample(dec3, scale_factor=2, mode='bilinear', align_corners=False),
            upsample(dec4, scale_factor=4, mode='bilinear', align_corners=False),
            upsample(dec5, scale_factor=8, mode='bilinear', align_corners=False),
        ], 1)

        segmentation = self.logit(self.final_do(hypercolumn))
        fused = torch.cat([
            hypercolumn,
            upsample(conv5_pool.unsqueeze(2).unsqueeze(3), scale_factor=128, mode='nearest')
        ], 1)
        fused = self.fusion(self.final_do(fused))
        return fused, classification, segmentation


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super().__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch: int, r: int = 16):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cse = self.cSE(x)
        sse = self.sSE(x)

        x = cse + sse

        return x


class DecoderBlockSCSE(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockSCSE, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
            SCSE(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    batch = torch.FloatTensor(12, 3, 128, 128).cuda()
    net = SeResNextUnet(pretrained=True).cuda()
    summary(net, input_size=(3, 128, 128))
    with Timer():
         print(net(batch)[0].size())
