import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm3d(out_ch*2),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm3d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose3d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_1=nn.Sequential(
            nn.Conv3d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.downsample= nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out1=self.Conv_BN_ReLU_1(x)
        # out2 = self.Conv_BN_ReLU_2(out1) + out1
        out2=self.Conv_BN_ReLU_2(out1)
        out_2=self.downsample(out2)
        return out2,out_2

class UNet(nn.Module):
    def __init__(self, ch_in=1):
        super(UNet, self).__init__()
        out_channels=[2**(i+4) for i in range(5)] #[16, 32, 64, 128, 256]
        #下采样
        self.d1=DownsampleLayer(1,out_channels[0])
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])
        #输出
        self.o=nn.Sequential(
            nn.Conv3d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(),
            nn.Conv3d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(),
            nn.Conv3d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        total_out = out
        return total_out


