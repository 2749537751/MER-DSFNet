
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
# from PVMamba import PVMamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from attition.Attition import CBAM
from model.backbone import build_backbone
import numpy as np

from module.fgbg_fusionmix import DSSFM




__all__ = ['UNet', 'NestedUNet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.conv1 = DeformConv2D(in_channels, middle_channels, 3, padding=1)


        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # self.conv2 = DeformConv2D(middle_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.router_pool = nn.AdaptiveAvgPool2d(1)
        self.router_fc = nn.Linear(input_channels, 2)
        # self.dssfm1_0 = DSSFM(in_channels=64,out_channels=64,dropout_rate=0.1,seg_dim=256,d_state=16)
        # self.dssfm2_0 = DSSFM(in_channels=128, out_channels=128, dropout_rate=0.2,seg_dim=128,d_state=16)
        # self.dssfm3_0 = DSSFM(in_channels=256, out_channels=256, dropout_rate=0.1,seg_dim=64,d_state=16)
        self.dssfm3_0 = DSSFM(in_channels=256, out_channels=256, dropout_rate=0.1)


        self.conv128_64 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv256_128 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv512_256 = nn.Conv2d(512, 256, kernel_size=1)


        # self.pvm0_0 = PVMamba(input_dim=3, output_dim=32)
        # self.pvm1_0 = PVMamba(input_dim=32, output_dim=64)
        # self.pvm2_0 = PVMamba(input_dim=64, output_dim=128)
        # self.pvm3_0 = PVMamba(input_dim=128, output_dim=256)
        # self.pvm4_0 = PVMamba(input_dim=256, output_dim=512)

    def forward(self, input):
        router_logits = self.router_fc(self.router_pool(input).flatten(1))
        router_probs = F.softmax(router_logits, dim=1)
        self.last_router_probs = router_probs
        topk_vals, topk_idx = torch.topk(router_probs, k=2, dim=1)
        router_weights = torch.zeros_like(router_probs).scatter(1, topk_idx, topk_vals)
        router_weights = router_weights / router_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        #U1
        x0_0 = self.conv0_0(input)  #   x0_0: torch.Size([4, 32, 512, 512])
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))  #   x1_0: torch.Size([4, 64, 256, 256])
        # print('x1_0:',x1_0.shape)
        # x1_0 = self.dssfm1_0(x1_0)   #   dssfm1_0: torch.Size([2, 64, 256, 256])
        # print('dssfm1_0:',x1_0.shape)
        x2_0 = self.conv2_0(self.pool(x1_0))  #   x2_0: torch.Size([4, 128, 128, 128])
        # print('x2_0:',x2_0.shape)
        # x2_00 =self.dssfm2_0(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))  #   x3_0: torch.Size([4, 256, 64, 64])
        # print('x3_0:',x3_0.shape)
        x3_00 =self.dssfm3_0(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))  #   x4_0: torch.Size([4, 512, 32, 32])
        # print('x4_0:',x4_0.shape)

        x3_1 = self.conv3_1(torch.cat([x3_00, self.up(x4_0)], 1))  #   x3_1: torch.Size([4, 256, 64, 64])
        x3_11 = self.dssfm3_0(x3_1)
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))  #   x2_2: torch.Size([4, 128, 128, 128])
        # x2_22 = self.dssfm2_0(x2_2)
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))  #   x1_3: torch.Size([4, 64, 256, 256])
        # x1_3 = self.dssfm1_0(x1_3)
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))  #   x0_4: torch.Size([4, 32, 512, 512])
        # print('x0_4:',x0_4.shape)
        expert1_output = self.final(x0_4)


        #U2
        X1_0 = self.conv1_0(self.pool(x0_4))  # X1_0: torch.Size([4, 64, 256, 256])
        X1_0 = self.conv128_64(torch.cat([X1_0, x1_3], 1))
        # X1_0 = self.dssfm1_0(X1_0)
        # print('x1_0:',x1_0.shape)
        X2_0 = self.conv2_0(self.pool(X1_0))  # X2_0: torch.Size([4, 128, 128, 128])
        X2_0 = self.conv256_128(torch.cat([X2_0, x2_2], 1))
        # X2_00 = self.dssfm2_0(X2_0)
        # print('x2_0:',x2_0.shape)
        X3_0 = self.conv3_0(self.pool(X2_0))  # X3_0: torch.Size([4, 256, 64, 64])
        X3_0 = self.conv512_256(torch.cat([X3_0, x3_11], 1))
        X3_00 = self.dssfm3_0(X3_0)
        # print('x3_0:',x3_0.shape)
        X4_0 = self.conv4_0(self.pool(X3_0))  # X4_0: torch.Size([4, 512, 32, 32])
        # print('x4_0:',x4_0.shape)

        X3_1 = self.conv3_1(torch.cat([X3_00, self.up(X4_0)], 1))  # X3_1: torch.Size([4, 256, 64, 64])
        # print('x3_1:',x3_1.shape)
        X2_2 = self.conv2_2(torch.cat([X2_0, self.up(X3_1)], 1))  # X2_2: torch.Size([4, 128, 128, 128])
        # print('x2_2:',x2_2.shape)
        X1_3 = self.conv1_3(torch.cat([X1_0, self.up(X2_2)], 1))  # X1_3: torch.Size([4, 64, 256, 256])
        # print('x1_3:',x1_3.shape)
        X0_4 = self.conv0_4(torch.cat([x0_0, self.up(X1_3)], 1))  # X0_4: torch.Size([4, 32, 512, 512])
        # print('x0_4:',x0_4.shape)
        expert2_output = self.final(X0_4)

        w1 = router_weights[:, 0].view(-1, 1, 1, 1)
        w2 = router_weights[:, 1].view(-1, 1, 1, 1)
        output = w1 * expert1_output + w2 * expert2_output
        return output
#                 #U1
#         x0_0 = self.conv0_0(input)  #   x0_0: torch.Size([4, 32, 512, 512])
#         # print('x0_0:',x0_0.shape)
#         x1_0 = self.conv1_0(self.pool(x0_0))  #   x1_0: torch.Size([4, 64, 256, 256])
#         # print('x1_0:',x1_0.shape)
#         x1_0 = self.dssfm1_0(x1_0)   #   dssfm1_0: torch.Size([2, 64, 256, 256])
#         # print('dssfm1_0:',x1_0.shape)
#         x2_0 = self.conv2_0(self.pool(x1_0))  #   x2_0: torch.Size([4, 128, 128, 128])
#         # print('x2_0:',x2_0.shape)
#         x2_0 =self.dssfm2_0(x2_0)
#         x3_0 = self.conv3_0(self.pool(x2_0))  #   x3_0: torch.Size([4, 256, 64, 64])
#         # print('x3_0:',x3_0.shape)
#         x3_0 =self.dssfm3_0(x3_0)
#         x4_0 = self.conv4_0(self.pool(x3_0))  #   x4_0: torch.Size([4, 512, 32, 32])
#         # print('x4_0:',x4_0.shape)

#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  #   x3_1: torch.Size([4, 256, 64, 64])
#         x3_1 = self.dssfm3_0(x3_1)
#         # print('x3_1:',x3_1.shape)
#         x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))  #   x2_2: torch.Size([4, 128, 128, 128])
#         x2_2 = self.dssfm2_0(x2_2)
#         # print('x2_2:',x2_2.shape)
#         x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))  #   x1_3: torch.Size([4, 64, 256, 256])
#         x1_3 = self.dssfm1_0(x1_3)
#         # print('x1_3:',x1_3.shape)
#         x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))  #   x0_4: torch.Size([4, 32, 512, 512])
#         # print('x0_4:',x0_4.shape)


#         #U2
#         X1_0 = self.conv1_0(self.pool(x0_4))  # X1_0: torch.Size([4, 64, 256, 256])
#         X1_0 = self.conv128_64(torch.cat([X1_0, x1_3], 1))
#         X1_0 = self.dssfm1_0(X1_0)
#         # print('x1_0:',x1_0.shape)
#         X2_0 = self.conv2_0(self.pool(X1_0))  # X2_0: torch.Size([4, 128, 128, 128])
#         X2_0 = self.conv256_128(torch.cat([X2_0, x2_2], 1))
#         X2_0 = self.dssfm2_0(X2_0)
#         # print('x2_0:',x2_0.shape)
#         X3_0 = self.conv3_0(self.pool(X2_0))  # X3_0: torch.Size([4, 256, 64, 64])
#         X3_0 = self.conv512_256(torch.cat([X3_0, x3_1], 1))
#         X3_0 = self.dssfm3_0(X3_0)
#         # print('x3_0:',x3_0.shape)
#         X4_0 = self.conv4_0(self.pool(X3_0))  # X4_0: torch.Size([4, 512, 32, 32])
#         # print('x4_0:',x4_0.shape)

#         X3_1 = self.conv3_1(torch.cat([X3_0, self.up(X4_0)], 1))  # X3_1: torch.Size([4, 256, 64, 64])
#         # print('x3_1:',x3_1.shape)
#         X2_2 = self.conv2_2(torch.cat([X2_0, self.up(X3_1)], 1))  # X2_2: torch.Size([4, 128, 128, 128])
#         # print('x2_2:',x2_2.shape)
#         X1_3 = self.conv1_3(torch.cat([X1_0, self.up(X2_2)], 1))  # X1_3: torch.Size([4, 64, 256, 256])
#         # print('x1_3:',x1_3.shape)
#         X0_4 = self.conv0_4(torch.cat([x0_0, self.up(X1_3)], 1))  # X0_4: torch.Size([4, 32, 512, 512])
#         # print('x0_4:',x0_4.shape)



        # return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output




    