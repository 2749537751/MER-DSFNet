import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torchvision import models
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
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
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=512, dim_head=64, dropout=0.,
                 emb_dropout=0.):
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


import torch
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # ======================
    # 参数设置
    # ======================
    batch_size = 4
    image_size = (64, 64)  # 输入图像尺寸 (H, W)
    patch_size = 8  # 每个Patch的尺寸
    channels = 32  # 输入通道数（需与测试输入匹配）
    dim = 32  # Transformer的隐藏层维度
    depth = 4  # Transformer的层数（减少深度以加快测试）
    heads = 8  # 注意力头数
    mlp_dim = 256  # MLP隐藏层维度
    dropout = 0.1
    emb_dropout = 0.1

    # ======================
    # 生成随机输入（模拟1024通道的32x32特征图）
    # ======================
    x = torch.randn(batch_size, channels, image_size[0], image_size[1]) * 2 - 1  # 范围[-1, 1]


    # ======================
    # 初始化ViT模型
    # ======================
    vit = ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    # ======================
    # 前向传播测试
    # ======================
    with torch.no_grad():
        output = vit(x)
    print("输入形状:", x.shape)  # 应为 (batch_size, channels, H, W)
    print("ViT输出形状:", output.shape)  # 应为 (batch_size, dim, H, W)

    # # ======================
    # # 可视化输入和输出（取第一个样本的第0通道）
    # # ======================
    # plt.figure(figsize=(12, 6))
    #
    # # 输入可视化
    # plt.subplot(1, 2, 1)
    # plt.imshow(x[0, 0].numpy(), cmap='viridis')  # 取第0通道
    # plt.title("Input (Channel 0)")
    # plt.colorbar()
    #
    # # 输出可视化（注意：输出通道数=dim=1024，这里取第0通道仅用于演示）
    # plt.subplot(1, 2, 2)
    # plt.imshow(output[0, 0].numpy(), cmap='viridis')  # 取第0通道
    # plt.title("Output (Channel 0)")
    # plt.colorbar()
    #
    # plt.tight_layout()
    # plt.show()

    # # ======================
    # # 可选：测试Transformer子模块
    # # ======================
    # print("\n测试Transformer子模块:")
    #
    # # 1. 测试Attention模块
    # print("\n--- Attention测试 ---")
    # attn = Attention(dim=dim, heads=heads, dim_head=dim // heads)
    # dummy_input = torch.randn(batch_size, seq_len=64, dim)  # 模拟序列输入
    # attn_output = attn(dummy_input)
    # print("Attention输入形状:", dummy_input.shape)
    # print("Attention输出形状:", attn_output.shape)  # 应为 (batch_size, seq_len, dim)
    #
    # # 2. 测试Transformer块
    # print("\n--- Transformer块测试 ---")
    # transformer_block = Transformer(dim=dim, depth=2, heads=heads, dim_head=dim // heads, mlp_dim=mlp_dim)
    # transformer_output = transformer_block(dummy_input)
    # print("Transformer输入形状:", dummy_input.shape)
    # print("Transformer输出形状:", transformer_output.shape)