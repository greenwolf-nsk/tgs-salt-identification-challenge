import torch
import torchvision
from torch import nn
from torch.nn.functional import upsample
import torch.nn.functional as F


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


class ConvReluConv(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = conv3x3(in_channels, middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
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


class DeeplySupervised34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 256),
        )
        self.fc = nn.Linear(512, 1)

        self.decoder5 = DecoderBlockSCSE(512 + 256, 512, 256)
        self.decoder4 = DecoderBlockSCSE(256 + 256, 512, 256)
        self.decoder3 = DecoderBlockSCSE(256 + 128, 256, 128)
        self.decoder2 = DecoderBlockSCSE(128 + 64, 128, 64)
        self.decoder1 = DecoderBlockSCSE(64, 64, 32)

        self.decoder5_conv = ConvReluConv(256, 32, 1)
        self.decoder4_conv = ConvReluConv(256, 32, 1)
        self.decoder3_conv = ConvReluConv(128, 32, 1)
        self.decoder2_conv = ConvReluConv(64, 32, 1)
        self.decoder1_conv = ConvReluConv(32, 32, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(1248, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self.logit = nn.Sequential(
            nn.Conv2d(736, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        e5_pool = nn.functional.avg_pool2d(e5, kernel_size=4).view(x.size(0), -1)
        classification = self.fc(e5_pool)

        center = self.center(e5)

        dec5 = self.decoder5(torch.cat([center, e5], 1))
        dec4 = self.decoder4(torch.cat([dec5, e4], 1))
        dec3 = self.decoder3(torch.cat([dec4, e3], 1))
        dec2 = self.decoder2(torch.cat([dec3, e2], 1))
        dec1 = self.decoder1(dec2)

        up_dec1 = dec1
        up_dec2 = upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False)
        up_dec3 = upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False)
        up_dec4 = upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False)
        up_dec5 = upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False)

        hypercolumn = torch.cat([up_dec1, up_dec2, up_dec3, up_dec4, up_dec5], 1)

        segmentation = [
            self.decoder5_conv(up_dec5),
            self.decoder4_conv(up_dec4),
            self.decoder3_conv(up_dec3),
            self.decoder2_conv(up_dec2),
            self.decoder1_conv(up_dec1),
        ]

        fused = torch.cat([
            hypercolumn,
            upsample(e5_pool.unsqueeze(2).unsqueeze(3), scale_factor=128, mode='nearest')
        ], 1)
        fused = self.fusion(fused)
        return fused, classification, segmentation


if __name__ == '__main__':
    batch = torch.FloatTensor(24, 3, 128, 128).cuda()
    net = DeeplySupervised34().cuda()
    print(net(batch)[1].size())
