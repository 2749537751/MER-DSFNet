
import torch.nn.functional as F

import torch.nn as nn
import torch

import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module.mamba_test import VSSBlock

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
"""Decouple Layer"""
class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        # f_uc = self.cbr_uc(x)
        # return f_fg, f_bg, f_uc
        return f_fg, f_bg

class CDFA(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1,attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.dcp = DecoupleLayer(in_c,dim)
    def forward(self, x,fg, bg):

        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted




def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=512, dim_head=64, dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)


        # 这里上采样倍数为8倍。为了保持和图中的feature size一样
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU())

    def forward(self, img):
        # 这里对应了图中的Linear Projection，主要是将图片分块嵌入，成为一个序列

        x = self.to_patch_embedding(img)


        b, n, _ = x.shape

        # 为图像切片序列加上索引
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)


        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 输入到Transformer中处理
        x = self.transformer(x)


        # delete cls_tokens, 输出前需要删除掉索引
        output = x[:, 1:, :]
        # print(output.size())
        output = self.out(output)


        # Transformer输出后，上采样到原始尺寸
        output = self.upsample(output)
        output = self.conv(output)

        return output









class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result



class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.group_num, -1)  # 替代 .view()
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias




__all__ = ['DSSFM']
    
    
    

#高级模块缝合：串行+融合+并行+门控权重

class DSSFM(nn.Module):
    def __init__(self, in_channels=32,out_channels=32,image_size=(64, 64), patch_size=8,dropout_rate=0.2):
        super(DSSFM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = ViT(image_size=image_size , patch_size=patch_size, channels=in_channels, dim=out_channels, depth=12, heads=4, mlp_dim=32,
                       dropout=0.1, emb_dropout=0.1).to(self.device)

        self.CBAM = CBAM(in_planes=out_channels).to(self.device)
        self.sigmod = nn.Sigmoid().to(self.device)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device)
        self.cdfa = CDFA(in_c=in_channels,out_c=out_channels).to(self.device)
        self.mamba = nn.Sequential(
            VSSBlock(hidden_dim=in_channels, attn_drop_rate=0.1, d_state=16),
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        ).to(self.device)
        
        # 添加 Dropout 层
        self.dropout_x1 = nn.Dropout2d(p=dropout_rate).to(self.device)  # 用于 x1（ViT 输出）
        self.dropout_x2 = nn.Dropout2d(p=dropout_rate).to(self.device)  # 用于 x2（VSSBlock 输出）
        self.dropout_fusion = nn.Dropout2d(p=dropout_rate).to(self.device)  # 用于 fusion_x

    def forward(self, x):
        # print('x:', x.shape)
        x=x.to(self.device)
        x1 = self.vit(x)  # C:32
        # x1 = self.dropout_x1(x1)  # 添加 Dropout
        # print('x1:', x1.shape)
        x2 = self.mamba(x)  # C:32
        # x2 = self.dropout_x2(x2)  # 添加 Dropout
        # print('x2:', x2.shape)
        a = self.sigmod(x1+x2)


        x3 = x1*a
        x4 = x2*(1-a)
        X3 = self.conv(x3)  #fg
        X4 = self.conv(x4)  #bg
        x5 = self.conv(x4+x3)  #mix=x
        fusion_x = self.cdfa(x5, X3, X4)
        # fusion_x = self.dropout_fusion(fusion_x)
        # x6 = x3+x4
        # print('fusion_x:', fusion_x.shape)
        # fusion_x = self.SEMAConv(fusion_x)
        out = self.CBAM(fusion_x) + x
        return out
if __name__ == '__main__':
    input = torch.randn(1,32,64,64)
    DSSFM_model = DSSFM(32,32)
    output = DSSFM_model(input)
    print('input:',input.shape)
    print('output:', output.shape)
    # if __name__ == '__main__':
    #     x = torch.randn(1,32,64,64)
    #     ema = SEMAConv(32)
    #     print('input:',x.shape)
    #     print('output:', ema(x).shape)

    # 举例2：通道拼接操作
    # class SEMAConv(nn.Module):
    #     def __init__(self, channels=32 ):
    #         super(SEMAConv, self).__init__()
    #         self.ScConv = ScConv(op_channel=channels)
    #         self.EMA = EMA(channels=channels)
    #         self.conv = nn.Conv2d(channels * 2 ,channels,kernel_size=1,stride=1)
    #     def forward(self,x):
    #         x1 = self.ScConv(x) #C:32
    #         print(x1.shape)
    #         x2 = self.EMA(x)   #C:32
    #         print(x2.shape)
    #         x = torch.cat([x1,x2],dim=1) #C:64
    #         print(x.shape)
    #         x = self.conv(x)
    #         return x
    # if __name__ == '__main__':
    #     x = torch.randn(1,32,64,64)
    #     ema = SEMAConv(32)
    #     print('input:',x.shape)
    #     print('output:', ema(x).shape)

    # 举例3：门控权重相乘操作
    # class SEMAConv(nn.Module):
    #     def __init__(self, channels=32 ):
    #         super(SEMAConv, self).__init__()
    #         self.ScConv = ScConv(op_channel=channels)
    #         self.EMA = EMA(channels=channels)
    #         self.sigmod = nn.Sigmoid()
    #         self.conv = nn.Conv2d(channels * 2 ,channels,kernel_size=1,stride=1)
    #     def forward(self,x):
    #         x1 = self.ScConv(x) #C:32
    #         x2 = self.EMA(x)   #C:32
    #         X1 = self.sigmod(x1)
    #         x = x1*x2
    #         return x
    # if __name__ == '__main__':
    #     x = torch.randn(1,32,64,64)
    #     ema = SEMAConv(32)
    #     print('input:',x.shape)
    #     print('output:', ema(x).shape)

    # 串行缝合：A+B+C
    # 融合模块缝合

    # 并行模块缝合：A->(A,B,C,D,E...)->A

    # 举例1：相加操作

    # 举例2：通道拼接操作

    # 举例3：门控权重相乘操作