import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from .blocks import upconv, Conv2dBlock, ResBlocks, outputConv

class SkyEncoder(nn.Module):
    def __init__(self, cin=3, cout=64, activ='relu'):
        super(SkyEncoder, self).__init__()
        self.conv1_1 = Conv2dBlock(cin, 32, 5, 1, 2, norm='in', activation=activ, pad_type='zero')
        # convb(cin, 32, k=5, stride=1, pad=2) # 32, 128, 3 -> 32, 128, 32
        self.res_block1 = ResBlocks(2, 32, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(32, 32, stride=1) # 32, 128, 32 -> 32, 128, 32
        self.conv1_2 = Conv2dBlock(32, 64, 3, 2, 1, norm='in', activation=activ, pad_type='zero')
        # convb(32, 64, k=3, stride=2, pad=1) # 32, 128, 32 -> 16, 64, 64
        self.conv2_1 = Conv2dBlock(64, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(64, 64, k=3, stride=1, pad=1) # 16, 64, 64 -> 16, 64, 64
        self.res_block2 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(64, 64, stride=1) # 16, 64, 64 -> 16, 64, 64
        self.conv2_2 = Conv2dBlock(64, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero')
        # convb(64, 128, k=3, stride=2, pad=1) # 16, 64, 64 -> 8, 32, 128
        self.conv3_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(128, 128, k=3, stride=1, pad=1) # 8, 32, 128 -> 8, 32, 128
        self.res_block3 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(128, 128, stride=1) # 8, 32, 128 -> 8, 32, 128
        self.conv3_2 = Conv2dBlock(128, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero')
        # convb(128, 128, k=3, stride=2, pad=1) # 8, 32, 128 -> 4, 16, 128
        self.conv4_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(128, 128, k=3, stride=1, pad=1) # 4, 16, 128 -> 4, 16, 128
        self.res_block4 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(128, 128, stride=1) # 4, 16, 128 -> 4, 16, 128
        self.conv4_2 = Conv2dBlock(128, 64, 3, 2, 1, norm='none', activation=activ, pad_type='zero')
        # convb(128, 64, k=3, stride=2, pad=1) # 4, 16, 128 -> 2, 8, 64
        self.fc = nn.Linear(1024, cout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        out = self.conv1_1(inputs)
        out = self.res_block1(out)
        out = self.conv1_2(out)
        out = self.conv2_1(out)
        out = self.res_block2(out)
        out = self.conv2_2(out)
        out = self.conv3_1(out)
        out = self.res_block3(out)
        out = self.conv3_2(out)
        out = self.conv4_1(out)
        out = self.res_block4(out)
        out = self.conv4_2(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

class SkyDecoder(nn.Module):
    def __init__(self, cin=64, cout=3, activ='relu'):
        super(SkyDecoder, self).__init__()
        self.fc = nn.Linear(cin, 1024)
        self.conv1 = Conv2dBlock(16, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(16, 64, k=3, stride=1, pad=1) # 4, 16, 16 -> 4, 16, 64
        self.res_block1 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(64, 64, stride=1) # 4, 16, 64 -> 4, 16, 64
        self.upconv1 = upconv(64, 64, k=3, stride=1, pad=1) # 4, 16, 64 -> 8, 32, 64
        self.conv2 = Conv2dBlock(64, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(64, 128, k=3, stride=1, pad=1) # 8, 32, 64 -> 8, 32, 128
        self.res_block2 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(128, 128, stride=1) # 8, 32, 128 -> 16, 64, 128
        self.upconv2 = upconv(128, 128, k=3, stride=1, pad=1) # 8, 32, 128 -> 16, 64, 128
        self.conv3 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(128, 64, k=3, stride=1, pad=1) # 16, 64, 128 -> 16, 64, 64
        self.res_block3 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(64, 64, stride=1) # 16, 64, 64 -> 16, 64, 64
        self.upconv3 = upconv(64, 64, k=3, stride=1, pad=1) # 16, 64, 64 -> 32, 128, 64
        self.conv4 = Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero')
        # convb(64, 32, k=3, stride=1, pad=1) # 32, 128, 64 -> 32, 128, 32
        self.res_block4 = ResBlocks(2, 32, norm='in', activation=activ, pad_type='zero')
        # BasicBlock(32, 32, stride=1) # 32, 128, 32 -> 32, 128, 32
        self.outputconv_1 = Conv2dBlock(32, 16, 3, 1, 1, norm='none', activation=activ, pad_type='zero')
        self.outputconv_2 = outputConv(16, cout) # 32, 128, 32 -> 32, 128, 3

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        out = self.fc(inputs)
        out = out.view(-1, 16, 4, 16)
        out = self.conv1(out)
        out = self.res_block1(out)
        out = self.upconv1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.upconv2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.upconv3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        out = self.outputconv_1(out)
        out = self.outputconv_2(out)

        return out