"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Sequential
from torch.nn.functional import upsample
from torch.nn import functional as F
from collections import OrderedDict

from torchsummary import summary

from modules.wider_resnet import WiderResNet


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
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

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


class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3, pretrained=False):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        super(TernausNetV2, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=0)
        if pretrained:
            state_dict = torch.load('wide_resnet38_ipabn_lr_256.pth.tar')['state_dict']
            state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final_do = nn.Dropout2d(p=0.5)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(self.final_do(dec1))


class WideUnet(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3, pretrained=False):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=0)
        if pretrained:
            state_dict = torch.load('wide_resnet38_ipabn_lr_256.pth.tar')['state_dict']
            state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final_do = nn.Dropout2d(p=0.25)
        self.final_hyper = nn.Conv2d(num_filters * 20, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        f = torch.cat([
            dec1,
            dec2,
            upsample(dec3, scale_factor=2, mode='bilinear', align_corners=False),
            upsample(dec4, scale_factor=4, mode='bilinear', align_corners=False),
            upsample(dec5, scale_factor=8, mode='bilinear', align_corners=False),
        ], 1)
        return self.final_hyper(self.final_do(f))


class WideUnetSE(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3, pretrained=False):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=0)
        if pretrained:
            state_dict = torch.load('wide_resnet38_ipabn_lr_256.pth.tar')['state_dict']
            state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final_do = nn.Dropout2d(p=0.5)
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec_se5(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.dec_se4(self.dec4(torch.cat([dec5, conv4], 1)))
        dec3 = self.dec_se3(self.dec3(torch.cat([dec4, conv3], 1)))
        dec2 = self.dec_se2(self.dec2(torch.cat([dec3, conv2], 1)))
        dec1 = self.dec_se1(self.dec1(torch.cat([dec2, conv1], 1)))

        return self.final(self.final_do(dec1))


if __name__ == '__main__':
    batch = torch.FloatTensor(1, 3, 128, 128).cuda()
    net = TernausNetV2(num_classes=2).cuda()
    summary(net, input_size=(3, 128, 128))
    print(net(batch).size())
